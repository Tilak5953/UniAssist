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

# 2. Stop and rebuild containers to guarantee fresh update
echo "Rebuilding UniAssist gateway and RAG containers..."
sudo docker compose stop app-service rag-service || true
sudo docker compose rm -f app-service rag-service || true
sudo docker compose build --no-cache app-service rag-service
sudo docker compose up -d --force-recreate app-service rag-service

# Re-ingest knowledge base
sleep 3
curl -s -X POST http://localhost:8001/ingest || true

# 3. Status
echo "=========================================================="
echo " UniAssist Updated & Live!"
sudo docker compose ps
echo "=========================================================="
