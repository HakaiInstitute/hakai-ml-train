#!/bin/bash
# Get instance ID, Instance AZ, Volume ID and Volume AZ
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
AWS_REGION=us-east-1
SPOT_FLEET_REQUEST_ID=$(aws ec2 describe-spot-instance-requests --region $AWS_REGION --filter "Name=instance-id,Values='$INSTANCE_ID'" --query "SpotInstanceRequests[].Tags[?Key=='aws:ec2spot:fleet-request-id'].Value[]" --output text)

# Install EFS Tools
yum install -y amazon-efs-utils

# Mount EFS filesystem
mkdir -p /dltraining
mount -t efs -o tls fs-4d8532f9:/ /dltraining
chown -R ec2-user: /dltraining/
cd /home/ec2-user/ || exit

# Change the docker image installation cache directory
# Sed script to add the new docker data-root to the docker.json config file
sed -i '0,/{/ s//{\n    "data-root": "\/dltraining\/docker-data",/' /etc/docker/daemon.json
systemctl daemon-reload
systemctl restart docker

# Get training code
git clone https://github.com/tayden/uav-classif.git
chown -R ec2-user: uav-classif
cd uav-classif/scripts/presence || exit
mkdir -p ./train_input/data
mount --bind /dltraining/presence/data ./train_input/data
mkdir -p ./train_output
mount --bind /dltraining/presence/train_output ./train_output

# Initiate training using the pytorch_36 conda environment
sudo -H -u ec2-user bash -c "source /home/ec2-user/anaconda3/bin/activate python3; bash ./build_and_train.sh"

# After training, clean up by cancelling spot requests and terminating itself
aws ec2 cancel-spot-fleet-requests --region $AWS_REGION --spot-fleet-request-ids "$SPOT_FLEET_REQUEST_ID" --terminate-instances