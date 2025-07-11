#!/bin/bash
set -e

echo "Updating package lists..."
sudo apt update

echo "Installing vim..."
sudo apt install -y vim

echo "Installing zellij..."
ZELLIJ_VERSION=$(curl -s https://api.github.com/repos/zellij-org/zellij/releases/latest | grep tag_name | cut -d '"' -f 4)
ZELLIJ_DEB="zellij-${ZELLIJ_VERSION#v}-amd64.deb"
ZELLIJ_URL="https://github.com/zellij-org/zellij/releases/download/${ZELLIJ_VERSION}/${ZELLIJ_DEB}"

echo "Downloading Zellij ${ZELLIJ_VERSION}..."
curl -LO "$ZELLIJ_URL"

echo "Installing Zellij..."
sudo dpkg -i "$ZELLIJ_DEB"

rm "$ZELLIJ_DEB"

echo "All done!"

