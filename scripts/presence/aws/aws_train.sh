# From tutorial https://aws.amazon.com/blogs/machine-learning/train-deep-learning-models-on-gpus-using-amazon-ec2-spot-instances/
# This file just details some commands that were used to set up the Spot instance training. It is not intended to be run but serve
# As a reference for what was done to create the EBS block and how to run the spot instances using the json config.

# Create a server
aws ec2 run-instances \
    --image-id ami-0dbb717f493016a1a \
    --count 1 \
    --instance-type m4.large \
    --key-name 'dl-training' \
    --subnet-id 'subnet-00926ee5fc42d2608' \
    --query "Instances[0].InstanceId" \
    --security-group-ids 'sg-0f0a49e25692b4784' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=DL-Config-Kelp}]'

# Create a volume
aws ec2 create-volume \
    --size 32 \
    --region 'us-east-1' \
    --availability-zone 'us-east-1b' \
    --volume-type gp2 \
    --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=DL-seagrass-checkpoints}]'

# Attach volume to server
aws ec2 attach-volume \
    --volume-id vol-078256099328edf0b \
    --instance-id i-0aed24eddbac99763 \
    --device /dev/sdf

# Upload data to the mounted EBS volume using scp

# SSH to server
ssh ec2-user@ec2-3-90-156-131.compute-1.amazonaws.com

# Create Spot fleet role
aws iam create-role \
     --role-name DL-Training-Spot-Fleet-Role \
     --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Sid":"","Effect":"Allow","Principal":{"Service":"spotfleet.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam attach-role-policy \
     --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole --role-name DL-Training-Spot-Fleet-Role

# Turn train script to base64 and attach to spot-fleet request
base64 mount-efs.sh -w0 | xclip -selection c

# Request the spot instances
aws ec2 request-spot-fleet --spot-fleet-request-config file://spot-fleet-config-m4-large.json
aws ec2 request-spot-fleet --spot-fleet-request-config file://spot-fleet-config-p3-2xlarge.json
aws ec2 request-spot-fleet --spot-fleet-request-config file://spot-fleet-config-p3-8xlarge.json

