# Description

## Local

Command to load data

### Command to load artefacts

```bash
p run python app/model.py
```

#### Command to Backend

```bash
p run uvicorn app.main:app
```

### Command to Front

```bash
p run python -m http.server 8080
```

## Container

### Backend


#### Build

```bash
docker run --env-file .env -d -p 80:80 fastapi-pet-classifier

```

#### Run Container

```bash
docker run --env-file .env -d -p 80:80 fastapi-pet-classifier
```

### Front

#### Build

```bash
docker build -f Dockerfile.front -t front-end-app .
```

#### Run

```bash
docker run -p 8080:80 front-end-app
```

## Compose


### Compose Build

```bash
docker-compose -f docker-compose.yml --env-file .env build --no-cache
```

### Compose Run

```bash
docker-compose -f docker-compose.yml --env-file .env up -d
```

## Compose Down

```bash
docker-compose -f docker-compose.yml down --volumes
```


## Push images to Repo

```bash
docker build -f Dockerfile.back -t ebagirov/lsml_backend:0.1 .
docker build -f Dockerfile.front -t ebagirov/lsml_frontend:0.1 .

docker push ebagirov/lsml_backend:0.1
docker push ebagirov/lsml_front:0.1
```
