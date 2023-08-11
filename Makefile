# PHONY are targets with no files to check, all in our case
.PHONY: build
.DEFAULT_GOAL := build

build:
	docker compose build
	docker compose down
	docker compose up -d
	docker exec -it oss_api bash
	