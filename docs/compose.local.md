# docker-compose.local.yml - Local Development Setup

The `docker-compose.local.yml` file is used to set up and run both the back-end and front-end services locally without using an external Docker registry. This file builds the Docker images directly from the `Dockerfile.back` and `Dockerfile.front` files in your project.

---

## Overview

The `docker-compose.local.yml` defines two services:

1. **`core_service` (Back-End)**:
   - Builds the back-end Docker image from `Dockerfile.back`.
   - Exposes the back-end service on port `80`.
   - Passes environment variables as build arguments for the Comet model.

2. **`front_service` (Front-End)**:
   - Builds the front-end Docker image from `Dockerfile.front`.
   - Exposes the front-end service on port `8080`.
   - Relies on the `core_service` to provide the `BACKEND_URL`.

---

## Prerequisites

1. Ensure Docker and Docker Compose are installed and running on your system.
2. Create a `.env` file in the project root to provide required environment variables:

   ```dotenv
   COMET_API_KEY=your-comet-api-key
   COMET_WORKSPACE=your-comet-workspace
   COMET_MODEL_NAME=your-model-name
   COMET_MODEL_NAME_VERSION=your-model-version
   COMET_MODEL_FILE=your-model-file-path
   ```
