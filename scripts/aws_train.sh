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
    --instance-id i-0d26889523a4df950 \
    --device /dev/sdf

# SSH to server
ssh ubuntu@ec2-54-174-122-132.compute-1.amazonaws.com
