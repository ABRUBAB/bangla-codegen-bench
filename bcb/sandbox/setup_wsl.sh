#!/usr/bin/env bash
# One-time setup of the WSL2 fallback sandbox (used when Docker is unavailable).
# Uses Ubuntu's own python3 + apt packages (PyPI is not needed).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends python3 python3-pytest python3-yaml python3-defusedxml python3-lxml \
  python3-bcrypt python3-cryptography python3-argon2 python3-passlib python3-numpy python3-requests \
  python3-markupsafe python3-jinja2 python3-bleach gzip tar zip unzip coreutils util-linux grep
id -u runner >/dev/null 2>&1 || useradd -m -u 1001 runner
mkdir -p /opt/bcb
chmod -R a+rX /opt/bcb
python3 -c "import pytest, yaml, defusedxml, lxml, bcrypt, cryptography, argon2, passlib, numpy, requests, markupsafe, jinja2, bleach; print('IMPORTS_OK', pytest.__version__)"
echo SETUP_OK
