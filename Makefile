
all: build_local
SHELL := /bin/bash


build:
	docker build --network host --compress -t ${DOCKER_NAMESPACE}/rxwgan:latest .
	docker build --network host --compress -t ${DOCKER_NAMESPACE}/rxwgan:${DOCKER_IMAGE_TAG} .

build_local:
	virtualenv -p python ${VIRTUALENV_NAMESPACE}
	source ${VIRTUALENV_NAMESPACE}/bin/activate && pip install poetry && poetry install

build_image:
	make build
	make push
	make pull


push:
	docker push ${DOCKER_NAMESPACE}/rxwgan:latest
	docker push ${DOCKER_NAMESPACE}/rxwgan:${DOCKER_IMAGE_TAG}

pull: 
	singularity pull docker://${DOCKER_NAMESPACE}/rxwgan:${DOCKER_IMAGE_TAG}
	mv rxwgan_${DOCKER_IMAGE_TAG}.sif ${SINGULARITY_IMAGES_REPO}

run:
	singularity run --nv --bind=/home:/home --writable-tmpfs ${SINGULARITY_IMAGES_REPO}/rxwgan_${DOCKER_IMAGE_TAG}.sif  

clean:
	docker system prune -a
	