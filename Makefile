
all: build_local
SHELL := /bin/bash



build_local:
	virtualenv -p python ${VIRTUALENV_NAMESPACE}
	source ${VIRTUALENV_NAMESPACE}/bin/activate && pip install -r requirements.txt

build_images:
	make build_base
	make build_sif


build_base:
	docker build --network host --build-arg  --compress -t ${DOCKER_NAMESPACE}/wgan:base -f Dockerfile .


build_sif:
	docker push ${DOCKER_NAMESPACE}/wgan:base
	singularity pull docker://${DOCKER_NAMESPACE}/wgan:base
	mv *.sif ${PROJECT_DIR}/images



run:
	singularity run --nv --bind=/home:/home  --bind=/mnt/brics_data:/mnt/brics_data --writable-tmpfs ${PROJECT_DIR}/images/wgan_base.sif  



clean:
	docker system prune -a
	