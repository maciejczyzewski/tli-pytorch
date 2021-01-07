#!/bin/sh -e
set -x

isort --recursive  --force-single-line-imports --apply *.py
autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place *.py --exclude=__init__.py
black *.py
isort --recursive --apply *.py
