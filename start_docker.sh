#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 <container_name> <image[:tag]> [extra-docker-run-args]" >&2
    exit 1
}

check_docker() {
    if ! command -v docker &>/dev/null; then
        echo "ERROR: Docker CLI not found. Please install Docker." >&2
        exit 2
    fi
    if ! docker info &>/dev/null; then
        echo "ERROR: Docker daemon not running or permission denied." >&2
        exit 2
    fi
}

build_default_args() {
    local CUR_PWD="/home/ccui1002/dev"

    DEFAULT_RUN_ARGS=(
        -it
        -d
        --privileged
        --network=host
        --device=/dev/kfd
        --device=/dev/dri
        --group-add video
        --cap-add=SYS_PTRACE
        --security-opt seccomp=unconfined
        -v /dev/infiniband:/dev/infiniband
        -v /home:/home
        -v /data:/data
        -v /work:/work
        -v /mnt:/mnt
        -v /DISK:/DISK
        -v /DISK2:/DISK2
        -v /DISK3:/DISK3
        -w "${CUR_PWD}"
        --shm-size=64G
        --ulimit memlock=-1
        --ulimit stack=67108864
    )
}

setup_ssh_config() {
    local cname="$1"

    docker exec "${cname}" bash -c '
        set -e
        SSH_DIR="$HOME/.ssh"
        CONFIG="$SSH_DIR/config"

        mkdir -p "$SSH_DIR"
        chmod 700 "$SSH_DIR"

        if ! grep -q "^Host github.com$" "$CONFIG" 2>/dev/null; then
            cat >> "$CONFIG" <<EOF

Host github.com
  HostName github.com
  User git
  IdentityFile /home/ccui1002/.ssh/id_ed25519
EOF
        fi

        chmod 600 "$CONFIG"
    '
}

main() {
    if [[ $# -lt 1 ]]; then
        usage
    elif [[ $# -eq 1 ]]; then
        IMAGE=$1
        CONTAINER_NAME="auto_$(date +%s%N | cut -c1-16)"
        echo "No container name provided. Using generated name: ${CONTAINER_NAME}"
        shift 1
        EXTRA_ARGS=()
    else
        CONTAINER_NAME=$1
        IMAGE=$2
        shift 2
        EXTRA_ARGS=("$@")
    fi

    check_docker
    build_default_args

    if docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container '${CONTAINER_NAME}' is already running."
        exit 0

    elif docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Starting existing container '${CONTAINER_NAME}'..."
        docker start "${CONTAINER_NAME}"
        echo "Container started successfully."
        exit 0

    else
        echo "Creating and starting new container '${CONTAINER_NAME}' from image '${IMAGE}'..."
        docker run "${DEFAULT_RUN_ARGS[@]}" "${EXTRA_ARGS[@]}" \
            --name "${CONTAINER_NAME}" \
            "${IMAGE}"

        setup_ssh_config "${CONTAINER_NAME}"

        echo "Container '${CONTAINER_NAME}' created and started successfully."
        exit 0
    fi
}

main "$@"
