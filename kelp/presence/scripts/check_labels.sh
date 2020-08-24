THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKING_DIR="/mnt/Z/kelp_presence_data"



# Mount the samba data server
#sudo mkdir -p /mnt/H
#sudo mount -t cifs -o user=taylor.denouden,domain=victoria.hakai.org //10.10.1.50/Geospatial /mnt/H

# Execute all following instructions in the uav-classif/deeplabv3/kelp directory
cd "$WORKING_DIR/train/y" || exit 1
python "$THIS_DIR/check_labels.py"

cd "$WORKING_DIR/eval/y" || exit 1
python "$THIS_DIR/check_labels.py"

