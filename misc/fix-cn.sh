#!/usr/bin/env bash
# fix-cn.sh — Fix Chinese garbled text on Linux
# Usage:
#   sudo bash fix-cn.sh            # 系统级修复（推荐，需 root）
#   bash fix-cn.sh --user-only     # 仅修复当前用户环境变量
#   LANG_CHOICE=zh_CN sudo bash fix-cn.sh   # 指定默认界面语言（zh_CN/en_US）

set -euo pipefail

USER_ONLY="${1:-}"
LANG_CHOICE="${LANG_CHOICE:-zh_CN}"  # zh_CN 或 en_US（界面语言）
NEED_SUDO=true
[[ "$USER_ONLY" == "--user-only" ]] && NEED_SUDO=false

# ---- helpers ----
have() { command -v "$1" &>/dev/null; }
infile_has() { grep -qE "$2" "$1" 2>/dev/null || return 1; }

ensure_sudo() {
  $NEED_SUDO || return 0
  if [[ $EUID -ne 0 ]]; then
    echo ">> 请以 root 或使用 sudo 运行（或改用 --user-only）。"
    exit 1
  fi
}

# ---- detect distro ----
ID=""; ID_LIKE=""
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
fi
ID="${ID:-}"; ID_LIKE="${ID_LIKE:-}"

is_deb()  { [[ "$ID" == "debian" || "$ID" == "ubuntu" || "$ID_LIKE" =~ (debian|ubuntu) ]]; }
is_rhel() { [[ "$ID" =~ (rhel|centos|rocky|almalinux|fedora) || "$ID_LIKE" =~ (rhel|fedora) ]]; }
is_arch() { [[ "$ID" == "arch" || "$ID_LIKE" =~ (arch) ]]; }

# ---- install locale & fonts ----
install_pkgs() {
  $NEED_SUDO || return 0
  echo ">> 安装/确认本地化与字体包 ..."
  if is_deb; then
    apt-get update -y
    DEBIAN_FRONTEND=noninteractive apt-get install -y locales language-pack-zh-hans fonts-noto-cjk
  elif is_rhel; then
    if have dnf; then PM=dnf; else PM=yum; fi
    $PM -y install glibc-langpack-zh glibc-langpack-en langpacks-zh_CN \
                   google-noto-sans-cjk-ttc google-noto-serif-cjk-ttc || true
    # 某些发行版字体包名为：noto-sans-cjk-fonts / noto-serif-cjk-fonts
    $PM -y install noto-sans-cjk-fonts noto-serif-cjk-fonts || true
  elif is_arch; then
    pacman -Sy --noconfirm glibc noto-fonts-cjk
  else
    echo "!! 未识别发行版，跳过包管理自动安装。请手动确保已安装 UTF-8 语言包与 Noto CJK 字体。"
  fi
}

# ---- generate locales ----
gen_locales() {
  $NEED_SUDO || return 0
  echo ">> 生成/启用 zh_CN.UTF-8 与 en_US.UTF-8 ..."
  if is_deb; then
    # 解注释 /etc/locale.gen
    sed -ri 's/^#\s*(zh_CN\.UTF-8)/\1/' /etc/locale.gen || true
    sed -ri 's/^#\s*(en_US\.UTF-8)/\1/' /etc/locale.gen || true
    # 若行不存在则追加
    infile_has /etc/locale.gen '^zh_CN\.UTF-8' || echo 'zh_CN.UTF-8 UTF-8' >> /etc/locale.gen
    infile_has /etc/locale.gen '^en_US\.UTF-8' || echo 'en_US.UTF-8 UTF-8' >> /etc/locale.gen
    locale-gen
    update-locale
  else
    # 通用 localedef（大多数系统可用）
    localedef -c -f UTF-8 -i zh_CN zh_CN.UTF-8   || true
    localedef -c -f UTF-8 -i en_US en_US.UTF-8   || true
  fi
}

# ---- set system-wide locale ----
set_system_locale() {
  $NEED_SUDO || return 0
  echo ">> 配置系统默认语言环境 ..."
  if is_deb; then
    # Debian/Ubuntu 使用 /etc/default/locale
    {
      echo "LANG=${LANG_CHOICE}.UTF-8"
      echo "LC_CTYPE=zh_CN.UTF-8"
      echo "LC_NUMERIC=en_US.UTF-8"
      echo "LC_TIME=en_US.UTF-8"
      echo "LC_COLLATE=C"
      echo "LC_MONETARY=zh_CN.UTF-8"
      echo "LC_MESSAGES=${LANG_CHOICE}.UTF-8"
      echo "LC_PAPER=zh_CN.UTF-8"
      echo "LC_NAME=zh_CN.UTF-8"
      echo "LC_ADDRESS=zh_CN.UTF-8"
      echo "LC_TELEPHONE=zh_CN.UTF-8"
      echo "LC_MEASUREMENT=zh_CN.UTF-8"
      echo "LC_IDENTIFICATION=zh_CN.UTF-8"
    } > /etc/default/locale
  else
    # RHEL/Fedora/Arch 使用 /etc/locale.conf
    {
      echo "LANG=${LANG_CHOICE}.UTF-8"
      echo "LC_CTYPE=zh_CN.UTF-8"
      echo "LC_TIME=en_US.UTF-8"
      echo "LC_COLLATE=C"
    } > /etc/locale.conf
  fi
}

# ---- set user locale (if --user-only or in addition to system) ----
set_user_locale() {
  echo ">> 配置当前用户的语言环境 (~/.profile) ..."
  local rc="$HOME/.profile"
  touch "$rc"
  if ! infile_has "$rc" '## locale fix start'; then
    {
      echo ''
      echo '## locale fix start'
      echo "export LANG=${LANG_CHOICE}.UTF-8"
      echo 'export LC_CTYPE=zh_CN.UTF-8'
      echo 'export LC_TIME=en_US.UTF-8'
      echo 'export LC_COLLATE=C'
      echo '## locale fix end'
    } >> "$rc"
  else
    # 更新已有块
    awk -v lang="${LANG_CHOICE}.UTF-8" '
      BEGIN{inblk=0}
      /## locale fix start/{inblk=1; print; print "export LANG="lang; print "export LC_CTYPE=zh_CN.UTF-8"; print "export LC_TIME=en_US.UTF-8"; print "export LC_COLLATE=C"; next}
      /## locale fix end/{inblk=0}
      !inblk{print}
    ' "$rc" > "$rc.tmp" && mv "$rc.tmp" "$rc"
  fi
}

# ---- verify ----
verify() {
  echo ">> 验证当前会话与字体（重新登录后系统级设置才会全局生效）"
  echo "---- locale 输出 ----"
  locale || true
  echo "---- 中文显示测试 ----"
  printf '中文测试：你好，世界！\n'
  echo "---- 字体（Noto CJK）检测 ----"
  if have fc-list; then
    fc-list | grep -i -E 'Noto.*(CJK|Sans CJK|Serif CJK)' | head -n 5 || echo "(未检测到 Noto CJK 字体条目)"
    fc-cache -f >/dev/null 2>&1 || true
  else
    echo "提示：未找到 fc-list（fontconfig）。可安装 fontconfig 以检测字体。"
  fi
  echo ">> 完成。若终端仍乱码，请将终端编码设为 UTF-8 并重新登录/重启终端。"
}

main() {
  ensure_sudo
  install_pkgs
  gen_locales
  $NEED_SUDO && set_system_locale
  set_user_locale
  verify
}

main "$@"

