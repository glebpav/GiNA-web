#!/bin/bash

# flask --app app:create_app run --debug --port 8080
waitress-serve --port=8080 --call 'app:create_app'