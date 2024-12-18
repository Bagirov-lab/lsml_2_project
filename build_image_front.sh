#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found! Please ensure it exists in the current directory."
    exit 1
fi

# Set default variables
IMAGE_NAME="ebagirov/lsml_frontend"
TAG="0.1" # Default tag
DOCKERFILE_PATH="./Dockerfile.front"
PUSH_IMAGE=false # Default value for pushing the image

# Parse script arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --push) PUSH_IMAGE=true ;; # Enable push when --push is passed
        --tag) TAG="$2"; shift ;; # Specify custom tag
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Remove existing image if it exists
if docker images | grep -q "$IMAGE_NAME\s*$TAG"; then
    echo "Removing existing Docker image: $IMAGE_NAME:$TAG"
    docker rmi -f $IMAGE_NAME:$TAG
fi

# Build the Docker image with build arguments
echo "Building Docker image: $IMAGE_NAME:$TAG using $DOCKERFILE_PATH"

docker build \
--no-cache \
--build-arg BACKEND_URL=$BACKEND_URL \
-t $IMAGE_NAME:$TAG -f $DOCKERFILE_PATH .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker image $IMAGE_NAME:$TAG built successfully."
    
    # Tag the image as 'latest'
    echo "Tagging $IMAGE_NAME:$TAG as $IMAGE_NAME:latest"
    docker tag $IMAGE_NAME:$TAG $IMAGE_NAME:latest
    
    # Push the image if the --push flag is set
    if [ "$PUSH_IMAGE" = true ]; then
        echo "Pushing $IMAGE_NAME:$TAG to the registry"
        docker push $IMAGE_NAME:$TAG
        
        echo "Pushing $IMAGE_NAME:latest to the registry"
        docker push $IMAGE_NAME:latest
        
        echo "Docker image $IMAGE_NAME:$TAG and $IMAGE_NAME:latest pushed successfully."
    else
        echo "Skipping image push as --push flag is not set."
    fi
else
    echo "Failed to build Docker image $IMAGE_NAME:$TAG."
    exit 1
fi