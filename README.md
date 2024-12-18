# Lovely Pets Project

Welcome to the **Lovely Pets** project! This repository contains a front-end and back-end application managed with Docker and Docker Compose for both local development and production environments.

---

## Table of Contents

- [Lovely Pets Project](#lovely-pets-project)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Scripts](#scripts)
    - [Build Scripts](#build-scripts)

---

## Project Structure

The repository is organized as follows:

.
├── Dockerfile.back        # Dockerfile for the back-end service
├── Dockerfile.front       # Dockerfile for the front-end service
├── build_image_back.sh    # Script to build the back-end Docker image
├── build_image_front.sh   # Script to build the front-end Docker image
├── run_image_back.sh      # Script to run the back-end container
├── run_image_front.sh     # Script to run the front-end container
├── docker-compose.local.yml # Compose file for local development
├── docker-compose.yml     # Compose file for production (prebuilt images)
├── docs/                  # Additional documentation
│   ├── backend.md
│   ├── frontend.md
│   ├── compose.local.md
│   ├── compose.md
│   └── notes.md
└── README.md              # This file

---

## Scripts

### Build Scripts

1. **`build_image_back.sh`**  
   Builds the back-end Docker image.  
   Usage:

   ```bash
   ./build_image_back.sh
   ./build_image_back.sh --push

2. build_image_front.sh

Builds the front-end Docker image.
Usage:

./build_image_front.sh
./build_image_front.sh --push --tag 1.0

Run Scripts

 1. run_image_back.sh
Runs the back-end container.
Usage:

./run_image_back.sh
./run_image_back.sh --tag 1.0

 2. run_image_front.sh
Runs the front-end container.
Usage:

./run_image_front.sh
./run_image_front.sh --tag 1.0 --port 8080

Docker Compose Files

Local Development

To run the back-end and front-end applications locally without an external Docker registry, use docker-compose.local.yml:

docker-compose -f docker-compose.local.yml up -d

 • Back-end: <http://localhost:80>
 • Front-end: <http://localhost:8080>

To stop the services:

docker-compose -f docker-compose.local.yml down

Production

To run the services using prebuilt images with the latest tag, use docker-compose.yml:

docker-compose -f docker-compose.yml up -d

 • Back-end: <http://localhost:80>
 • Front-end: <http://localhost:8080>

To stop the services:

docker-compose -f docker-compose.yml down

Usage

Step-by-Step Workflow

 1. Build the Images Locally:
 • Build the back-end:

./build_image_back.sh

 • Build the front-end:

./build_image_front.sh

 2. Run Services Locally:
Use the docker-compose.local.yml file:

docker-compose -f docker-compose.local.yml up -d

 3. Push Images to Registry:
Tag and push the images:

./build_image_back.sh --push
./build_image_front.sh --push --tag 1.0

 4. Run Production Services:
Use the docker-compose.yml file with prebuilt images:

docker-compose -f docker-compose.yml up -d

Documentation

Detailed documentation for each part of the project is available in the docs/ folder:
 • Back-End: docs/backend.md
 • Front-End: docs/frontend.md
 • Local Development: docs/compose.local.md
 • Production: docs/compose.md
 • Additional Notes: docs/notes.md

Notes
 • Ensure Docker and Docker Compose are installed on your system.
 • Use the .env file to provide required environment variables such as BACKEND_URL, COMET_API_KEY, etc.
 • For production, make sure the images are pushed to a Docker registry and tagged as latest.
