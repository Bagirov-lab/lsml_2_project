# Front-End Application

## build_image_front.sh - Build Docker Image

The `build_image_front.sh` script automates the process of building a Docker image for the front-end application.

---

### Prerequisites

1. Ensure Docker is installed and running on your system.
2. Create a `.env` file in the same directory as the script. The `.env` file should define the following environment variable:
   - `BACKEND_URL` (the URL of your back-end service).
3. Make the script executable:

   ```bash
   chmod +x build_image_front.sh
   ```

## run_image_front.sh - Run Docker Container

The `run_image_front.sh` script automates the process of running a Docker container for the front-end application. It ensures a clean environment by stopping and removing any existing container with the same name before starting a new one.

---

### Prerequisites

1. Ensure the Docker image has already been built using the `build_image_front.sh` script.
2. Make the script executable:

   ```bash
   chmod +x run_image_front.sh
   ```
