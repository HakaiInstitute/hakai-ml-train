# From tutorial https://aws.amazon.com/blogs/machine-learning/train-deep-learning-models-on-gpus-using-amazon-ec2-spot-instances/

# Create a server
aws ec2 run-instances \
    --image-id ami-0dbb717f493016a1a \
    --count 1 \
    --instance-type m4.xlarge \
    --key-name 'dl-training' \
    --subnet-id 'subnet-25a3790e' \
    --query "Instances[0].InstanceId" \
    --security-group-ids 'sg-0f0a49e25692b4784' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=dl-config}]'

# Create a volume
aws ec2 create-volume \
    --size 100 \
    --region 'us-east-1' \
    --availability-zone 'us-east-1d' \
    --volume-type gp2 \
    --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=DL-datasets-checkpoints}]'

# Attach volume to server
aws ec2 attach-volume \
    --volume-id vol-026e6c7d749e50474 \
    --instance-id i-0017cc77db0160262 \
    --device /dev/sdf

# SSH to server
ssh ubuntu@ec2-54-158-207-105.compute-1.amazonaws.com

# Create Spot fleet role
aws iam create-role \
     --role-name DL-Training-Spot-Fleet-Role \
     --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Sid":"","Effect":"Allow","Principal":{"Service":"spotfleet.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam attach-role-policy \
     --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole --role-name DL-Training-Spot-Fleet-Role

# Turn train script to base64 and attach to spot-fleet request
USER_DATA=`base64 user-data-script.sh -w0`
sed -i '' "s|base64_encoded_bash_script|$USER_DATA|g" spot_fleet_config.json