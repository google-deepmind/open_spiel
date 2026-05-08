#!/bin/bash
# setup_env.sh - Automates the Jules-Web Sandbox setup

echo "Updating system and installing GNUbg..."
sudo apt-get update
sudo apt-get install -y gnubg

echo "Verifying installation..."
gnubg-cli --version

echo "Setup complete. Please provide your automation_key.json to authenticate."
