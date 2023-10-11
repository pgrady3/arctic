import argparse
import json
import os.path
import os.path as op
import random
import sys
from glob import glob

import cv2
import torch
from easydict import EasyDict
from loguru import logger
import numpy as np
import torch
from PIL import Image
import common.viewer as viewer_utils
from common.mesh import Mesh
from common.viewer import ViewerData

# sys.path = ["."] + sys.path

from common.body_models import construct_layers
from common.viewer import ARCTICViewer
from src.mesh_loaders.arctic import construct_hand_meshes, construct_object_meshes, construct_smplx_meshes
from pathlib import Path
import pickle
import shutil
from src.utils.interfield import compute_dist_mano_to_obj
from common.transforms import transform_points, solve_rigid_tf_np
from common.rot import batch_rodrigues, rotation_matrix_to_angle_axis


SAVE_ROOT = 'data/export'


def json_write(path, data, auto_mkdir=False):
    if auto_mkdir:
        mkdir(path, cut_filename=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def pkl_write(path, data, auto_mkdir=False):
    if auto_mkdir:
        mkdir(os.path.dirname(path))

    with open(path, 'wb') as file_handle:
        pickle.dump(data, file_handle, protocol=pickle.HIGHEST_PROTOCOL)


def mkdir(path, cut_filename=False):
    if cut_filename:
        path = os.path.dirname(os.path.abspath(path))
    Path(path).mkdir(parents=True, exist_ok=True)


def get_tform_from_pointclouds(from_points, to_points):
    ideal_rot, ideal_trans = solve_rigid_tf_np(from_points, to_points)
    ideal_tform = np.eye(4)
    ideal_tform[:3, :3] = ideal_rot
    ideal_tform[:3, 3] = ideal_trans.squeeze()

    return ideal_tform


def get_mano_rot_and_trans(camera_hand_verts, world2cam, mano_pose, mano_beta, world_mano_rot, world_mano_trans, mano_layer):
    rotmat = batch_rodrigues(torch.tensor(world_mano_rot).unsqueeze(0))
    mano_tform = np.eye(4)
    mano_tform[:3, :3] = rotmat.squeeze().numpy()
    mano_tform[:3, 3] = world_mano_trans

    new_tform = np.matmul(world2cam, mano_tform)
    new_rot, _ = cv2.Rodrigues(new_tform[:3, :3])
    ideal_rot = new_rot.squeeze()

    out_fixedrot = mano_layer(
        hand_pose=torch.tensor(mano_pose).unsqueeze(0),
        betas=torch.tensor(mano_beta).unsqueeze(0),
        global_orient=torch.tensor(ideal_rot, dtype=torch.float32).unsqueeze(0),
        transl=torch.tensor([0, 0, 0], dtype=torch.float32).unsqueeze(0),
    ).vertices.numpy().squeeze()

    ideal_trans = camera_hand_verts[0, :] - out_fixedrot[0, :]

    # local_verts_trymano = mano_layer(
    #     hand_pose=torch.tensor(mano_pose).unsqueeze(0),
    #     betas=torch.tensor(mano_beta).unsqueeze(0),
    #     global_orient=torch.tensor(ideal_rot, dtype=torch.float32).unsqueeze(0),
    #     transl=torch.tensor(ideal_trans, dtype=torch.float32).unsqueeze(0),
    # ).vertices.numpy().squeeze()
    #
    # err = np.mean(local_verts_trymano - camera_hand_verts)
    # print('mean fitting err', err)
    return ideal_rot, ideal_trans


def construct_meshes(seq_p, layers, use_mano, use_object, use_smplx, no_image, use_distort, view_idx, subject_meta):
    # load
    data = np.load(seq_p, allow_pickle=True).item()
    cam_data = data["cam_coord"]
    data_params = data["params"]
    # unpack
    subject = seq_p.split("/")[-2]
    seq_name = seq_p.split("/")[-1].split(".")[0]
    obj_name = seq_name.split("_")[0]

    num_frames = cam_data["verts.right"].shape[0]

    # camera intrinsics
    if view_idx == 0:
        K = torch.FloatTensor(data_params["K_ego"][0].copy())
    else:
        K = torch.FloatTensor(
            np.array(subject_meta[subject]["intris_mat"][view_idx - 1])
        )

    # image names
    vidx = np.arange(num_frames)
    image_idx = vidx + subject_meta[subject]["ioi_offset"]
    imgnames = [
        f"./data/arctic_data/data/images/{subject}/{seq_name}/{view_idx}/{idx:05d}.jpg"
        for idx in image_idx
    ]

    # construct meshes
    vis_dict = {}
    if use_mano:
        right, left = construct_hand_meshes(cam_data, layers, view_idx, use_distort)
        vis_dict["right"] = right
        vis_dict["left"] = left
    if use_smplx:
        smplx_mesh = construct_smplx_meshes(cam_data, layers, view_idx, use_distort)
        vis_dict["smplx"] = smplx_mesh
    if use_object:
        obj = construct_object_meshes(cam_data, obj_name, layers, view_idx, use_distort)
        vis_dict["object"] = obj

    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )

    num_frames = len(imgnames)
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = np.eye(3)
    Rt[:, 1:3, :3] *= -1.0

    im = Image.open(imgnames[0])
    cols, rows = im.size
    if no_image:
        imgnames = None

    viewer_data = ViewerData(Rt, K, cols, rows, imgnames)

    for hand in ['left', 'right']:
        save_folder = os.path.join(SAVE_ROOT, subject, seq_name, f'{view_idx}_{hand}')
        if os.path.exists(os.path.join(save_folder, '{:05d}.pkl'.format(0))):
            continue

        hand_letter = hand[0]

        # Calcualte distance from each hand point to closest object
        knn_hand_verts = torch.tensor(data['cam_coord'][f'verts.{hand}'][:, view_idx, :, :])
        knn_obj_verts = torch.tensor(data['cam_coord'][f'verts.object'][:, view_idx, :, :])
        knn_dists, knn_idx = compute_dist_mano_to_obj(knn_hand_verts, knn_obj_verts, None, -1, 100)

        for i in range(num_frames):
            pkl_path = os.path.join(save_folder, '{:05d}.pkl'.format(i))
            json_path = os.path.join(save_folder, '{:05d}.json'.format(i))
            img_path = os.path.join(save_folder, '{:05d}.jpg'.format(i))

            if view_idx == 0:
                world2cam = np.array(data['params']['world2ego'][i, :, :])
            elif 1 <= view_idx <= 8:
                world2cam = np.array(subject_meta[subject]['world2cam'][view_idx - 1])

            save_dict = dict()
            save_dict['intrinsics'] = K.numpy()
            save_dict['image_size'] = [rows, cols]
            # save_dict['extrinsics'] = Rt[i, :, :]
            save_dict['hand_verts'] = vis_dict[hand]['v3d'][i, :, :]
            save_dict['hand_faces'] = vis_dict[hand]['f3d']
            save_dict['hand_joints'] = data['cam_coord'][f'joints.{hand}'][i, view_idx, :, :]
            save_dict['hand_dist'] = knn_dists[i, :].numpy()

            save_dict['mano_pose'] = data['params'][f'pose_{hand_letter}'][i, :]
            save_dict['mano_beta'] = data['params'][f'shape_{hand_letter}'][i, :]

            solved_rot, solved_trans = get_mano_rot_and_trans(save_dict['hand_verts'],
                                                            world2cam,
                                                            save_dict['mano_pose'],
                                                            save_dict['mano_beta'],
                                                            data['params'][f'rot_{hand_letter}'][i, :],
                                                            data['params'][f'trans_{hand_letter}'][i, :],
                                                            global_layers[hand])

            save_dict['mano_rot'] = solved_rot
            save_dict['mano_trans'] = solved_trans
            save_dict['hand'] = hand

            pkl_write(pkl_path, save_dict, auto_mkdir=True)

            # for key in save_dict.keys():    # Convert all torch tensors to numpy for serialization
            #     if isinstance(save_dict[key], np.ndarray):
            #         save_dict[key] = save_dict[key].tolist()
            #
            # json_write(json_path, save_dict, auto_mkdir=True)

            shutil.copy(imgnames[i], img_path)  # TODO: Copy the image file. Will eventually make this a crop

    return meshes, viewer_data


class DataViewer(ARCTICViewer):
    def __init__(self, render_types=["rgb", "depth", "mask"], interactive=True, size=(2024, 2024)):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = construct_layers(dev)
        super().__init__(render_types, interactive, size)

    def load_data(self, seq_p, use_mano, use_object, use_smplx, no_image, use_distort, view_idx, subject_meta):
        # from src.mesh_loaders.arctic import construct_meshes

        batch = construct_meshes(seq_p, self.layers, use_mano, use_object, use_smplx, no_image, use_distort, view_idx, subject_meta)
        self.check_format(batch)
        return batch


def main():
    with open("./data/arctic_data/data/meta/misc.json", "r") as f:
        subject_meta = json.load(f)

    # args = parse_args()
    random.seed(1)
    headless = False
    object = True
    mano = True
    smplx = False
    no_image = False
    distort = False
    print('DISTORT IS FALSE!!-------------------------------------------------------')

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    viewer = DataViewer(interactive=not headless, size=(2024, 2024))

    all_processed_seqs = glob("./outputs/processed_verts/seqs/*/*.npy")

    for seq_idx, seq_p in enumerate(all_processed_seqs):
        for view_idx in range(1, 9):
            print(f"Rendering seq#{seq_idx+1}, seq: {seq_p}, view: {view_idx}")
            seq_name = seq_p.split("/")[-1].split(".")[0]
            sid = seq_p.split("/")[-2]
            out_name = f"{sid}_{seq_name}_{view_idx}"

            batch = viewer.load_data(seq_p, mano, object, smplx, no_image, distort, view_idx, subject_meta)
            # viewer.render_seq(batch, out_folder=op.join("render_out", out_name))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    global_layers = construct_layers('cpu')

    main()
