#!/bin/bash

# Configuration values
IMAGE_NAME="yolov8_docker"  # Set the docker image name
IMAGE_TAG="1.0.0"           # Set the docker image tag
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
WORKING_DIR="/home/app"
LOCAL_INFERENCE_RESULTS_DIR="/path/to/local/inference/results"  # Set the local inference results directory
LOCAL_DATA_DIR="/path/to/local/data"                            # Set the local data directory
DOCKER_INFERENCE_RESULTS_DIR="${WORKING_DIR}/runs"
DOCKER_DATA_DIR="${WORKING_DIR}/data"

# Stop and remove any previous containers running from the given image
echo "Stopping and removing any existing containers for the image ${FULL_IMAGE_NAME}..."
CONTAINER_IDS=$(docker ps -a -q --filter ancestor=${FULL_IMAGE_NAME})

if [ -n "$CONTAINER_IDS" ]; then
  docker stop $CONTAINER_IDS
  docker rm $CONTAINER_IDS
  echo "Stopped and removed containers: $CONTAINER_IDS"
else
  echo "No containers found for image ${FULL_IMAGE_NAME}."
fi

# Remove the previous Docker image if it exists
echo "Removing the Docker image ${FULL_IMAGE_NAME} if it exists..."
IMAGE_IDS=$(docker images -q ${FULL_IMAGE_NAME})

if [ -n "$IMAGE_IDS" ]; then
  docker rmi $IMAGE_IDS
  echo "Removed image: $IMAGE_IDS"
else
  echo "No image found with the name ${FULL_IMAGE_NAME}."
fi

# Build the docker image from the Dockerfile
echo "Building the Docker image ${FULL_IMAGE_NAME}..."
docker build -t ${FULL_IMAGE_NAME} .

# Run the docker image
echo "Running the Docker image ${FULL_IMAGE_NAME}..."
docker run -d --ipc=host --gpus all -p 8080:8080 -v $LOCAL_INFERENCE_RESULTS_DIR:$DOCKER_INFERENCE_RESULTS_DIR -v $LOCAL_DATA_DIR:$DOCKER_DATA_DIR ${FULL_IMAGE_NAME}

echo "Docker container is up and running."