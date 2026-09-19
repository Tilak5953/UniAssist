#!/bin/bash
# ================================================================
# UniAssist - AWS EC2 One-Click Deployment Script
# Tested on: Ubuntu 22.04 / 24.04 LTS (t2.micro / t3.small / t3.medium)
# ================================================================

set -e

echo "=========================================================="
echo " Starting UniAssist Deployment on AWS EC2"
echo "=========================================================="

# 1. Update system packages
echo "Updating apt repositories..."
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# 2. Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo usermod -aG docker $USER
    echo "Docker installed successfully."
fi

# 3. Setup Swap for low-RAM AWS instances (Crucial for t2.micro with 1GB RAM)
if [ ! -f /swapfile ]; then
    echo "Configuring 2GB Swap space for low RAM AWS instance..."
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "Swap configured successfully."
fi

# 4. Build and Start Services with Docker Compose
echo "Building and launching UniAssist containers..."
sudo docker compose up -d --build

# 5. Pull ultra-lightweight models into Ollama container
echo "Waiting for Ollama container to become ready..."
sleep 8

echo "Pulling models into Ollama container..."
sudo docker exec uniassist-ollama ollama pull qwen2.5:0.5b
sudo docker exec uniassist-ollama ollama pull tinyllama

echo "=========================================================="
echo " UniAssist Deployment Completed Successfully!"
echo " Portal is live at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
echo "=========================================================="
