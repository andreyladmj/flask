#!/usr/bin/env bash
export FLASK_APP=app.py
python -m flask db upgrade
python -m flask db migrate
python -m flask filldb
echo "Run application"
python app.py
