# https://cloud.google.com/vertex-ai/docs/training/create-custom-container?hl=fr
LOCATION='location to host the image'
PROJECT_ID='GCP project id' # can be found out by running gcloud config list project --format "value(core.project)"
REPO_NAME='repository name in Artifact Registry'
IMAGE_NAME='image name'
VERSION='latest'

IMAGE_URI=${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${VERSION}

# go the location where the Dockerfileis located; then run
docker build ./ -t ${IMAGE_URI}

docker push ${IMAGE_URI}