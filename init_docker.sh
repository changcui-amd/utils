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

for pkg in vim curl tar jq; do
    if ! command -v $pkg &>/dev/null; then
        echo "Installing $pkg..."
        $INSTALL_CMD $pkg
    fi
done
