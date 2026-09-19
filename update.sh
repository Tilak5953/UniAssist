#!/bin/bash
# ================================================================
# UniAssist - Clean Update & Rebuild Script for AWS EC2
# ================================================================

set -e

echo "=========================================================="
echo " Updating UniAssist from GitHub repository..."
echo "=========================================================="

# 1. Force sync with origin/main to discard any local conflicts
git fetch origin main
git reset --hard origin/main

# 2. Stop and remove old gateway container to guarantee fresh rebuild
echo "Rebuilding UniAssist gateway container..."
sudo docker compose stop app-service || true
sudo docker compose rm -f app-service || true
sudo docker compose build --no-cache app-service
sudo docker compose up -d --force-recreate app-service

# 3. Status
echo "=========================================================="
echo " UniAssist Updated & Live!"
sudo docker compose ps
echo "=========================================================="
