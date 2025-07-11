#!/bin/bash
set -e

if command -v apt &>/dev/null; then
    PKG_MANAGER="apt"
    UPDATE_CMD="sudo apt update"
    INSTALL_CMD="sudo apt install -y"
elif command -v yum &>/dev/null; then
    PKG_MANAGER="yum"
    UPDATE_CMD="sudo yum makecache"
    INSTALL_CMD="sudo yum install -y"
else
    echo "Unsupported package manager. This script supports apt or yum only."
    exit 1
fi

echo "Updating package lists using $PKG_MANAGER..."
$UPDATE_CMD

echo "Installing vim..."
$INSTALL_CMD vim

for pkg in curl tar; do
    if ! command -v $pkg &>/dev/null; then
        echo "Installing $pkg..."
        $INSTALL_CMD $pkg
    fi
done

echo "Installing zellij to ~/.local/bin..."

ZELLIJ_VERSION=$(curl -s https://api.github.com/repos/zellij-org/zellij/releases/latest | grep tag_name | cut -d '"' -f 4)
ZELLIJ_TAR="zellij-x86_64-unknown-linux-musl.tar.gz"
ZELLIJ_URL="https://github.com/zellij-org/zellij/releases/download/${ZELLIJ_VERSION}/${ZELLIJ_TAR}"

INSTALL_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_DIR"

echo "Downloading $ZELLIJ_TAR..."
curl -LO "$ZELLIJ_URL"

echo "Extracting..."
tar -xzf "$ZELLIJ_TAR" zellij

mv zellij "$INSTALL_DIR"

rm "$ZELLIJ_TAR"

if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$HOME/.bashrc"
    echo "Added ~/.local/bin to PATH. Please reload your shell or run: source ~/.bashrc"
fi

echo "Zellij installed to $INSTALL_DIR/zellij"
echo "All done!"
