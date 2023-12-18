
export DOCKER_NAMESPACE=jodafons
export DOCKER_IMAGE_NAME=rxwgan
export DOCKER_IMAGE_TAG='1.0.0'
export SINGULARITY_IMAGES_REPO=/mnt/cern_data/images
export VIRTUALENV_NAMESPACE="rxwgan-env"


export REPO_BASEPATH=`pwd`
export BRICS_STORAGE=/home/brics/public/brics_data


# maestro
export DATABASE_SERVER_URL=$POSTGRES_SERVER_URL
export PGADMIN_DEFAULT_EMAIL=$POSTMAN_SERVER_EMAIL_FROM
export PGADMIN_DEFAULT_PASSWORD=$POSTMAN_SERVER_EMAIL_PASSWORD
export POSTMAN_SERVER_EMAIL_TO="jodafons@lps.ufrj.br"


[ -d ${VIRTUALENV_NAMESPACE} ] && source ${VIRTUALENV_NAMESPACE}/bin/activate
