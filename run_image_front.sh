#!/bin/bash

# Set default values
CONTAINER_NAME="lsml_frontend_container"
IMAGE_NAME="ebagirov/lsml_frontend:latest"
PORT=80 # Default port

# Parse script arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift ;; # Specify custom port
        --tag) IMAGE_NAME="ebagirov/lsml_frontend:$2"; shift ;; # Use specific image tag
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if the container exists and remove it
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping and removing existing container '$CONTAINER_NAME'..."
    docker stop $CONTAINER_NAME > /dev/null
    docker rm $CONTAINER_NAME > /dev/null
    echo "Container '$CONTAINER_NAME' removed."
fi

# Run a new container
echo "Starting new container '$CONTAINER_NAME' with image '$IMAGE_NAME' on port $PORT..."
docker run --name $CONTAINER_NAME -d -p $PORT:80 $IMAGE_NAME

if [ $? -eq 0 ]; then
    echo "Container '$CONTAINER_NAME' is now running. Access it at http://localhost:$PORT"
else
    echo "Failed to start container '$CONTAINER_NAME'."
    exit 1
fi