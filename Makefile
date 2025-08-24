.PHONY: build up logs stop restart down testnet


build:
docker compose build newrealtrading


up:
docker compose up -d newrealtrading


logs:
docker compose logs -f --tail=300 newrealtrading


stop:
docker compose stop newrealtrading || true


restart:
docker compose restart newrealtrading


down:
docker compose down newrealtrading || true


testnet:
docker compose up -d --build newrealtrading_testnet