#!/bin/bash

# Drop existing container (if any) and run the new one
if [ "$(docker ps -aq -f name=lsml_backend_container)" ]; then
    echo "Stopping and removing existing container 'lsml_backend_container'..."
    docker stop lsml_backend_container > /dev/null
    docker rm lsml_backend_container > /dev/null
    echo "Container 'lsml_backend_container' removed."
fi

# Run the new container
echo "Running new container 'lsml_backend_container'..."
docker run --name lsml_backend_container -d -p 80:80 \
ebagirov/lsml_backend:latest