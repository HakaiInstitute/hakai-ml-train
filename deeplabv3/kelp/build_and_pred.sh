# Get the path to this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Build the docker image
docker build --file ../Dockerfile --compress --tag tayden/deeplabv3-kelp ../..

# Get infile and outfile from first two args
infile=$1
outfile=$2
weights=$3
#infile="/home/tadenoud/PycharmProjects/uav-classif/deeplabv3/kelp/train_input/data/segmentation/mcnaughton_small.tif"
#outfile="/home/tadenoud/PycharmProjects/uav-classif/deeplabv3/kelp/train_output/segmentation/test.tif"
#weights="/home/tadenoud/PycharmProjects/uav-classif/deeplabv3/kelp/train_input/weights.pt"

outdir="$(dirname $outfile)"
outname="$(basename $outfile)"

# Run the docker image and bind data
docker run -it --rm \
-v "$DIR/train_output":/opt/ml/output \
-v "$outdir":/opt/segmentation/output \
--mount type=bind,source="$infile",target=/opt/segmentation/input/image.tif,readonly \
--mount type=bind,source="$weights",target=/opt/ml/input/weights.pt,readonly \
--user "$(id -u):$(id -g)" \
--ipc host \
--gpus all \
--name kelp-pred \
deeplabv3/kelp pred "$outname"