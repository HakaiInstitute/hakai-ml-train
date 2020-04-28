# Build the docker image
docker pull tayden/deeplabv3-kelp:v1.0.0

# Get infile and outfile from first two args
infile=$(realpath $1)
outfile=$(realpath $2)
weights=$(realpath $3)

outdir=$(dirname "$outfile")
outname=$(basename "$outfile")

# Run the docker image and bind data
docker run -it --rm \
-v "$outdir":/opt/segmentation/output \
--mount type=bind,source="$infile",target=/opt/segmentation/input/image.tif,readonly \
--mount type=bind,source="$weights",target=/opt/ml/input/weights.pt,readonly \
--user "$(id -u):$(id -g)" \
--ipc host \
--gpus all \
--name kelp-pred \
deeplabv3/kelp pred "$outname"