#!/bin/bash
set -e

# 检查系统使用的是 apt 还是 yum
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

# 更新软件包索引
echo "Updating package lists using $PKG_MANAGER..."
$UPDATE_CMD

# 安装 vim
echo "Installing vim..."
$INSTALL_CMD vim

# 安装 curl（如未安装）
if ! command -v curl &>/dev/null; then
    echo "Installing curl..."
    $INSTALL_CMD curl
fi

# 安装 zellij（从 GitHub 获取最新 release）
echo "Installing zellij..."
ZELLIJ_VERSION=$(curl -s https://api.github.com/repos/zellij-org/zellij/releases/latest | grep tag_name | cut -d '"' -f 4)
ZELLIJ_DEB="zellij-${ZELLIJ_VERSION#v}-amd64.deb"
ZELLIJ_URL="https://github.com/zellij-org/zellij/releases/download/${ZELLIJ_VERSION}/${ZELLIJ_DEB}"

echo "Downloading Zellij ${ZELLIJ_VERSION}..."
curl -LO "$ZELLIJ_URL"

echo "Installing Zellij..."
sudo dpkg -i "$ZELLIJ_DEB" || {
    echo "dpkg failed. Trying fallback with rpm conversion (requires 'alien')..."
    $INSTALL_CMD alien
    sudo alien -r "$ZELLIJ_DEB"
    RPM_FILE="${ZELLIJ_DEB%.deb}.rpm"
    sudo rpm -i "$RPM_FILE"
}

# 清理
rm -f "$ZELLIJ_DEB" "${RPM_FILE:-}"

echo "Installation complete!"

