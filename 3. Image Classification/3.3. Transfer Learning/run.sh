sudo docker run     --gpus all    -it     --rm     --shm-size 8G     --volume="$(pwd):$(pwd)"     --workdir="$(pwd)"     pytorch_custom:1.2

python train.py --use_gpu 0 --use_cores 16 --init_lr 1e-3 --batch_size 64 --max_epochs 100 --architecture VGG16 --model_name VGG16_with_BN
python train.py --use_gpu 1 --use_cores 16 --init_lr 1e-3 --batch_size 64 --max_epochs 100 --architecture resnet-18 --model_name resnet_18_without_pretrained_model
python train.py --use_gpu 2 --use_cores 16 --init_lr 1e-3 --batch_size 64 --max_epochs 100 --architecture resnet-18 --model_name resnet_18 --pretrained True
python train.py --use_gpu 3,4 --use_cores 16 --init_lr 1e-3 --batch_size 32 --max_epochs 100 --architecture densenet-161 --model_name densenet_161 --pretrained True

tensorboard --logdir=./expeirments/logs

