
all: build_local
SHELL := /bin/bash



build_local:
	virtualenv -p python ${VIRTUALENV_NAMESPACE}
	source ${VIRTUALENV_NAMESPACE}/bin/activate && pip install poetry && poetry install

build_images:
	make build_base
	make build_prod 


build_base:
	docker build --network host --build-arg  --compress -t ${DOCKER_NAMESPACE}/wgan:base -f base.Dockerfile .


build_prod:
	docker build --network host --build-arg DOCKER_NAMESPACE=${DOCKER_NAMESPACE} --build-arg GIT_SOURCE_COMMIT_ARG=$(git rev-parse HEAD) --compress -t ${DOCKER_NAMESPACE}/wgan:prod -f prod.Dockerfile .


build_base_sif:
	docker push ${DOCKER_NAMESPACE}/wgan:base
	singularity pull docker://${DOCKER_NAMESPACE}/wgan:base
	mv *.sif ${PROJECT_DIR}/images


build_prod_sif:
	docker push ${DOCKER_NAMESPACE}/wgan:prod
	singularity pull docker://${DOCKER_NAMESPACE}/wgan:prod
	mv *.sif ${PROJECT_DIR}/images


run:
	singularity run --nv --bind=/home:/home  --bind=/mnt/brics_data:/mnt/brics_data --writable-tmpfs ${PROJECT_DIR}/images/wgan_base.sif  

clean:
	docker system prune -a
	