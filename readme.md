
# Local
Command to load data

```
p run python app/model.py
```

Command to run server
```
p run uvicorn app.main:app
```

Run Web-App
```
p run python -m http.server 8080
````

# Container Core

## Core
Run Container

```
docker run --env-file .env -d -p 80:80 fastapi-pet-classifier
```

## Front

### Build
```
docker build -f Dockerfile.front -t front-end-app .
```

### Run

```
docker run -p 8080:80 front-end-app
```
Look over http://127.0.0.1:8080

# Compose 
Bild
```
docker-compose -f docker-compose.yml --env-file .env build --no-cache
```

Run
```
docker-compose -f docker-compose.yml --env-file .env up -d
```

Down
```
docker-compose -f docker-compose.yml down --volumes
```