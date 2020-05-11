#!/usr/bin/env bash
docker-compose stop
docker-compose up -d loadbalancer
docker-compose scale server=4
docker-compose stop loadbalancer
docker-compose rm loadbalancer
docker-compose up -d --no-deps loadbalancer