# Base Python image
FROM python:3.10

# Set working directory
WORKDIR /code

# Install Poetry
RUN pip install poetry==1.8.4

# Accept CI_JOB_TOKEN as a build argument
ARG COMET_API_KEY
ARG COMET_WORKSPACE
ARG COMET_MODEL_NAME
ARG COMET_MODEL_NAME_VERSION

# Create Environment
COPY ./pyproject.toml ./poetry.lock /code/
RUN poetry install

# Copy API
COPY ./app /code/app

# Dowload Model
RUN poetry run python app/model.py

# Default command
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]