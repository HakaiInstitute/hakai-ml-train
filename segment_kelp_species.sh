# Get infile, outfile, weights from args
in_file=$(realpath "$1")
out_file=$(realpath "$2")
weight_file=$(realpath "$3")

in_dir=$(dirname "$in_file")
out_dir=$(dirname "$out_file")
weight_dir=$(dirname "$weight_file")

# Run the docker image and bind data
docker run -it --rm \
-v "$in_dir":"$in_dir" \
-v "$out_dir":"$out_dir" \
-v "$weight_dir":"$weight_dir" \
--user "$(id -u):$(id -g)" \
--ipc host \
--gpus all \
--name kelp-pred-species \
tayden/deeplabv3-kelp-species:latest pred --seg_in="$in_file" --seg_out="$out_file" --weights="$weight_file"