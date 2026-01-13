#!/bin/bash

git pull || true
uv sync --all-extras --quiet --dev --frozen