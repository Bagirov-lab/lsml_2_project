
Command to run server
```
p run uvicorn app.main:app
```

Command to load data

```
p run python app/model.py

```

Run Container

```
docker run --env-file .env -d -p 80:80 fastapi-pet-classifier
```

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