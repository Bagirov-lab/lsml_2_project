#!/bin/bash

set -e  # Exit immediately on error

# Redirect output to a log file
LOG_FILE="script.log"
exec > >(tee -i "$LOG_FILE") 2>&1

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found! Please ensure it exists in the current directory."
    exit 1
fi

# Ensure essential environment variables are defined
required_env_vars=("COMET_API_KEY" "COMET_WORKSPACE" "COMET_MODEL_NAME" "COMET_MODEL_NAME_VERSION" "COMET_MODEL_FILE")
for var in "${required_env_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: Environment variable $var is not set. Please check your .env file."
        exit 1
    fi
done

# Set variables
IMAGE_NAME="ebagirov/lsml_backend"
TAG="0.1"
DOCKERFILE_PATH="./Dockerfile.back"
PUSH_IMAGE=false # Default value for pushing the image
DRY_RUN=false

# Parse script arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --push) PUSH_IMAGE=true ;;
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# If dry-run, exit early
if [ "$DRY_RUN" = true ]; then
    echo "Dry run enabled. Exiting without executing Docker commands."
    exit 0
fi

# Remove existing image if it exists
if docker images | grep -q "$IMAGE_NAME\s*$TAG"; then
    echo "Removing existing Docker image: $IMAGE_NAME:$TAG"
    docker rmi -f $IMAGE_NAME:$TAG
fi

# Build the Docker image with build arguments
echo "Building Docker image: $IMAGE_NAME:$TAG using $DOCKERFILE_PATH"

echo "COMET_MODEL_FILE: $COMET_MODEL_FILE"


docker build \
--no-cache \
--build-arg COMET_API_KEY=$COMET_API_KEY \
--build-arg COMET_WORKSPACE=$COMET_WORKSPACE \
--build-arg COMET_MODEL_NAME=$COMET_MODEL_NAME \
--build-arg COMET_MODEL_NAME_VERSION=$COMET_MODEL_NAME_VERSION \
--build-arg COMET_MODEL_FILE=$COMET_MODEL_FILE \
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