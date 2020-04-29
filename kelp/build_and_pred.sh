# Get the path to this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Build the docker image
docker build --file ../Dockerfile --compress --tag tayden/deeplabv3-kelp ../..

# Make output dirs
mkdir -p "./train_output/segmentation"

# Run the docker image and bind data
docker run -it --rm \
-v "$DIR/train_input":/opt/ml/input \
-v "$DIR/train_output":/opt/ml/output \
--user "$(id -u):$(id -g)" \
--ipc host \
--gpus all \
--name kelp-pred \
tayden/deeplabv3-kelp:latest pred "/opt/ml/input/data/segmentation/mcnaughton_small.tif" \
  "/opt/ml/output/segmentation/mcnaughton_small_kelp.tif" "/opt/ml/output/weights/deeplabv3_kelp_200421.pt"