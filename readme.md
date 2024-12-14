# Local

Command to load data

## Command to load artefacts

```bash
p run python app/model.py
```

## Command to run server

```bash
p run uvicorn app.main:app
```

## Command to run web-app

Run Web-App

```bash
p run python -m http.server 8080
````

# Container Core

## Core

Run Container

```bash
docker run --env-file .env -d -p 80:80 fastapi-pet-classifier
```

## Front

### Build

```bash
docker build -f Dockerfile.front -t front-end-app .
```

### Run

```bash
docker run -p 8080:80 front-end-app
```

# Compose


## Build

```bash
docker-compose -f docker-compose.yml --env-file .env build --no-cache
```

## Run

```bash
docker-compose -f docker-compose.yml --env-file .env up -d
```

## Down

```bash
docker-compose -f docker-compose.yml down --volumes
```
