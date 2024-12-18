# docker-compose.yml - Production Setup

The `docker-compose.yml` file is used to run both the back-end and front-end services using prebuilt Docker images that are assumed to be available in a Docker registry. This setup is ideal for production environments where images are already built and pushed to the registry.

---

## Overview

The `docker-compose.yml` defines two services:

1. **`core_service` (Back-End)**:
   - Uses a prebuilt Docker image for the back-end (`ebagirov/lsml_backend:latest`).
   - Exposes the back-end service on port `80`.

2. **`front_service` (Front-End)**:
   - Uses a prebuilt Docker image for the front-end (`ebagirov/lsml_frontend:latest`).
   - Exposes the front-end service on port `8080`.
   - Relies on the `core_service` for back-end communication.

---

## Prerequisites

1. Ensure Docker and Docker Compose are installed and running on your system.
2. Push the required Docker images to a Docker registry:
   - Back-End:

     ```bash
     docker push ebagirov/lsml_backend:latest
     ```

   - Front-End:

     ```bash
     docker push ebagirov/lsml_frontend:latest
     ```

---

## Services

### 1. `core_service` - Back-End

- **Image**:
  - Uses the prebuilt image `ebagirov/lsml_backend:latest`.
- **Ports**:
  - Exposes the back-end service on port `80`.
- **Network**:
  - Connected to the shared `lovely-pets-net` network.

### 2. `front_service` - Front-End

- **Image**:
  - Uses the prebuilt image `ebagirov/lsml_frontend:latest`.
- **Ports**:
  - Exposes the front-end service on port `8080`.
- **Network**:
  - Connected to the shared `lovely-pets-net` network.
- **Dependencies**:
  - Depends on the `core_service` to ensure the back-end is running before starting.

---

## Shared Network

- **Name**: `lovely-pets-net`
- **Type**: `bridge`
- **Purpose**: Allows communication between `core_service` and `front_service`.

---

## Usage

1. **Run Services**:
   Use the following command to start both services:

   ```bash
   docker-compose -f docker-compose.yml up -d
   ```
