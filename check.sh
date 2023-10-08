#! /bin/bash

set -e

black --line-length 200 *.py
isort *.py