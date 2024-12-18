# Environment Variables

This document explains the required environment variables, their usage, and where they are needed in the project.

---

## Overview of Environment Variables

The following environment variables are required to configure and build the back-end and front-end services.

| Variable Name              | Description                                      | Used In                                       |
|----------------------------|--------------------------------------------------|-----------------------------------------------|
| `COMET_API_KEY`            | API key for Comet ML integration.               | [Dockerfile.back](../Dockerfile.back)          |
| `COMET_WORKSPACE`          | Comet ML workspace name.                        | [Dockerfile.back](../Dockerfile.back)          |
| `COMET_MODEL_NAME`         | Name of the model to download.                  | [Dockerfile.back](../Dockerfile.back)          |
| `COMET_MODEL_NAME_VERSION` | Version of the model to download.               | [Dockerfile.back](../Dockerfile.back)          |
| `COMET_MODEL_FILE`         | File path where the model will be saved.        | [Dockerfile.back](../Dockerfile.back)          |
| `BACKEND_URL`              | URL of the back-end service for the front-end.  | [Dockerfile.back](../Dockerfile.back)          |

---

## Where Variables Are Used

### Back-End Service

- **Dockerfile**: [Dockerfile.back](../Dockerfile.back)
- **Docker Compose**: [docker-compose.local.yml](../docker-compose.local.yml)
- Environment variables for the back-end are passed as build arguments:
  - `COMET_API_KEY`
  - `COMET_WORKSPACE`
  - `COMET_MODEL_NAME`
  - `COMET_MODEL_NAME_VERSION`
  - `COMET_MODEL_FILE`

### Front-End Service

- **Dockerfile**: [Dockerfile.front](../Dockerfile.front)
- **Docker Compose**: [docker-compose.yml](../docker-compose.yml)
- Environment variable for the front-end:
  - `BACKEND_URL` (points to the back-end service URL).

---

## Prebuilt Images and [docker-compose.yml](../docker-compose.yml)

The [docker-compose.yml](../docker-compose.yml) file uses prebuilt images (`latest` tags) for both the back-end and front-end services. When running with [docker-compose.yml](../docker-compose.yml), you **do not need to rebuild images locally**.

---

## Summary of Usage

| File                             | Builds Images?            | Environment Variables Needed? |
|----------------------------------|---------------------------|--------------------------------|
| `Dockerfile.back`                | Yes                      | `COMET_API_KEY`, `COMET_WORKSPACE`, `COMET_MODEL_NAME`, `COMET_MODEL_NAME_VERSION`, `COMET_MODEL_FILE` |
| `Dockerfile.front`               | Yes                      | `BACKEND_URL`                 |
| `docker-compose.local.yml`       | Yes (Build Locally)       | Yes (from `.env` file)         |
| `docker-compose.yml`             | No (Uses Prebuilt Images) | No (Already Configured)        |

---

## Example `.env` File

Create a `.env` file in the root directory with the following content:

```dotenv
COMET_API_KEY=your-comet-api-key
COMET_WORKSPACE=your-comet-workspace
COMET_MODEL_NAME=your-model-name
COMET_MODEL_NAME_VERSION=your-model-version
COMET_MODEL_FILE=your-model-file-path
BACKEND_URL=http://localhost:80
```

An example file, [.env.example](../.env.example), has been provided for your convenience. Copy it as .env and update the values:

```bash
cp .env.example .env
```
