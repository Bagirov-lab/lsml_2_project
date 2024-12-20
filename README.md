# Lovely Pets Project 🐾

Welcome to the **Lovely Pets Project**! This repository contains a full-stack application, including a **back-end** and a **front-end** service, all managed using Docker and Docker Compose for local development and production.

---

## Table of Contents 📖

- [Lovely Pets Project 🐾](#lovely-pets-project-)
  - [Table of Contents 📖](#table-of-contents-)
  - [Project Overview 🚀](#project-overview-)
  - [Prerequisites ⚙️](#prerequisites-️)
  - [How to Use 🛠️](#how-to-use-️)
    - [Run Full Application Locally](#run-full-application-locally)
    - [Set Up ENV variables file](#set-up-env-variables-file)
    - [Build and Run Back-End](#build-and-run-back-end)
    - [Build and Run Front-End](#build-and-run-front-end)
  - [Documentation 📚](#documentation-)
  - [Project Structure 🗂️](#project-structure-️)
  - [License 📄](#license-)
  - [Feedback 💬](#feedback-)

---

## Project Overview 🚀

The Lovely Pets Project consists of two main components:

1. **Back-End Service**:
   - Provides API endpoints and model inference logic.
   - Built using Python and FastAPI.
   - Integrated with the Comet ML service to load models.

2. **Front-End Service**:
   - A static front-end served using NGINX.
   - Fetches back-end data via a configurable `BACKEND_URL`.

Both services are containerized with **Docker** and orchestrated using **Docker Compose** for development and production environments.

For more details, see [docs/project.md](docs/project.md).

---

## Prerequisites ⚙️

Ensure the following tools are installed on your system:

1. [Docker](https://www.docker.com/)
2. [Docker Compose](https://docs.docker.com/compose/)
3. Only Local Development additionaly:
   1. [Python 3.10+](https://www.python.org/)
   2. [Poetry](https://python-poetry.org/) (for local development)

## How to Use 🛠️

### Run Full Application Locally

Run Full Application in Production

To run the services using prebuilt images (pushed to Docker registry):

```bash
docker-compose -f docker-compose.yml up -d
```

For more details, see [docs/compose.md](docs/compose.md).

To run the back-end and front-end services together without an external Docker registry, you will need to supply `.env`.
Read more details about this file in see [docs/env_file.md](docs/env_file.md). After this you can run:

```bash
docker-compose -f docker-compose.local.yml up -d
```

- Back-End: `http://localhost:80`
- Front-End: `http://localhost:8080`

For more details, see [docs/compose.local.md](docs/compose.local.md).

### Set Up ENV variables file

Create a `.env` file in the root directory with the required environment variables:

```dotenv
COMET_API_KEY=your-comet-api-key
COMET_WORKSPACE=your-comet-workspace
COMET_MODEL_NAME=your-model-name
COMET_MODEL_NAME_VERSION=your-model-version
COMET_MODEL_FILE=your-model-file-path
BACKEND_URL=http://localhost:80
```

For more details, see [docs/env_file.md](docs/env_file.md).

### Build and Run Back-End

1. Build the Back-End Image:

    ```bash
    ./build_image_back.sh
    ```

2. Run the Back-End Container:

    ```bash
    ./run_image_back.sh
    ```

For more details, see [docs/backend.md](docs/backend.md).

### Build and Run Front-End

1. Build the Front-End Image:

    ```bash
    ./build_image_front.sh
    ```

2. Run the Front-End Container:

    ```bash
    ./run_image_front.sh
    ```

For more details, see [docs/frontend.md](docs/frontend.md).

## Documentation 📚

The repository includes detailed documentation for each part of the project:

- **Back-End**: [docs/backend.md](docs/backend.md)
- **Front-End**: [docs/frontend.md](docs/frontend.md)
- **Local Development**: [docs/compose.local.md](docs/compose.local.md)
- **Production Deployment**: [docs/compose.md](docs/compose.md)
- **Additional Notes**: [docs/notes.md](docs/notes.md)

## Project Structure 🗂️

```bash
.
├── app/                         # Back-end application code
│   ├── comet.py                 # Comet ML integration
│   ├── main.py                  # FastAPI entrypoint
│   ├── model.py                 # Model logic
│   └── train.py                 # Model training script
├── web-app/                     # Front-end static files
│   ├── example_photo.jpg        # Example photo for the application
│   ├── index.html               # Main HTML file
│   ├── script.js                # JavaScript logic for front-end
│   └── style.css                # Stylesheet for the front-end
├── docs/                        # Documentation files
│   ├── backend.md               # Back-End Documentation
│   ├── frontend.md              # Front-End Documentation
│   ├── compose.local.md         # Local Development Details
│   ├── compose.md               # Production Deployment Details
│   └── notes.md                 # Additional Notes
├── build_image_back.sh          # Script to build back-end image
├── build_image_front.sh         # Script to build front-end image
├── run_image_back.sh            # Script to run back-end container
├── run_image_front.sh           # Script to run front-end container
├── docker-compose.local.yml     # Compose file for local development
├── docker-compose.yml           # Compose file for production
├── Dockerfile.back              # Dockerfile for back-end
├── Dockerfile.front             # Dockerfile for front-end
├── pyproject.toml               # Poetry dependencies
├── .env.example                 # Example environment variables file
├── LICENSE                      # Project license
└── README.md                    # Project README
```

## License 📄

This project is licensed under the [MIT License](LICENSE).

## Feedback 💬

If you encounter any issues or have suggestions, feel free to open an issue or contribute to the repository. We appreciate your feedback!
