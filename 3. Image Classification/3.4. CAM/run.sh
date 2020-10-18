sudo docker run     --gpus all    -it     --rm     --shm-size 8G     --volume="$(pwd):$(pwd)"     --workdir="$(pwd)"     pytorch_custom:1.2

python train.py --use_gpu 0 --use_cores 16 --init_lr 1e-3 --batch_size 64 --max_epochs 100
python example_cam.py

tensorboard --logdir=./expeirments/tensorboard

