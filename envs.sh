# docker
export DOCKER_NAMESPACE=jodafons

# virtal env
export VIRTUALENV_NAMESPACE="wgan-env"

# git control
export GIT_SOURCE_COMMIT=$(git rev-parse HEAD)

# lab storage environs
export PROJECT_DIR=/mnt/brics_data

# output 
export TARGET_DIR=$PWD/targets

# data input
export DATA_DIR=$PROJECT_DIR/datasets

# mlflow tracking
export TRACKING_DIR=$PROJECT_DIR/tracking

# repo
export REPO_DIR=$PWD

# maestro environs
export DATABASE_SERVER_URL=$POSTGRES_SERVER_URL




# export to singularity
export SINGULARITYENV_DOCKER_NAMESPACE=${DOCKER_NAMESPACE}
export SINGULARITYENV_DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME}
export SINGULARITYENV_DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG}
export SINGULARITYENV_VIRTUALENV_NAMESPACE=${VIRTUALENV_NAMESPACE}
export SINGULARITYENV_PROJECT_DIR=${PROJECT_DIR}
export SINGULARITYENV_TARGET_DIR=${TARGET_DIR}
export SINGULARITYENV_DATA_DIR=${DATA_DIR}
export SINGULARITYENV_TACKING_DIR=${TRACKING_DIR}
export SINGULARITYENV_REPO_DIR=${REPO_DIR}
export SINGULARITYENV_DATABASE_SERVER_URL=$POSTGRES_SERVER_URL
export SINGULARITYENV_GIT_SOURCE_COMMIT=${GIT_SOURCE_COMMIT}
