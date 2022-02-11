# Install EFS Tools
yum install -y amazon-efs-utils

# Mount EFS filesystem
mkdir -p /data
mount -t efs -o tls fs-4d8532f9:/ /data