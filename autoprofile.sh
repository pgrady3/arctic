python -m cProfile -o profile.prof "$@"

read -p "Do you want to visualize the generated results?"

snakeviz profile.prof