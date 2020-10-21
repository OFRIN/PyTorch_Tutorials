# 1. Make docker image to run demo.ipynb.
docker build -t customized_pytorch:1.6.0 .

# 2. Run docker image
docker run \
    --gpus all\
    -it \
    --rm \
    -p 8888:8888 \
    --shm-size 8G \
    --volume="$(pwd):$(pwd)" \
    --volume="/media/sanghyun/Data/DB/Recon_DB/:../Recon_DB/" \
    --workdir="$(pwd)" \
    customized_pytorch:1.6.0

