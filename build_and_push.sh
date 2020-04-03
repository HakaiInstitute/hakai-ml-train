aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 612958249722.dkr.ecr.us-east-1.amazonaws.com/deeplabv3/kelp-train

docker build -t deeplabv3/kelp-train .

docker tag deeplabv3/kelp-train:latest 612958249722.dkr.ecr.us-east-1.amazonaws.com/deeplabv3/kelp-train:latest

docker push 612958249722.dkr.ecr.us-east-1.amazonaws.com/deeplabv3/kelp-train:latest