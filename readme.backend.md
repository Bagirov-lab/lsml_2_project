# Backend-End Application

## Build

### [build_image_back.sh](build_image_back.sh) - Backend Service

This script automates the process of building and optionally pushing a Docker image. It uses a `.env` file to load required environment variables for the build process.

#### Prerequisites

1. Ensure you have Docker installed and running on your system.
2. Place a `.env` file in the same directory as the script. The `.env` file should define the following environment variables:
   - `COMET_API_KEY`
   - `COMET_WORKSPACE`
   - `COMET_MODEL_NAME`
   - `COMET_MODEL_NAME_VERSION`
3. Make the script executable:

    ```bash
    chmod +x build_image_back.sh
    ```

4. Run the script:

   ```bash
   ./build_image_back.sh --push
   ```

## Run

### [build_image_back](build_image_back.sh) - Backend Service

The `run_image_back.sh` script automates the process of running a Docker container for the application. It ensures a clean environment by stopping and removing any existing container with the same name before starting a new one.

---

#### Prerequisites

1. Ensure the Docker image has already been built using the `build_image_back.sh` script.
2. Docker must be installed and running on your system.
3. Make the script executable:

   ```bash
   chmod +x run_image_back.sh
   ```
4. Run the script:

   ```bash
   ./run_image_back.sh
   ```
