#!/usr/bin/env bash
#
# start_docker.sh
# 用途：根据容器名和镜像名启动（或创建并启动）Docker 容器。
# 传参：
#   $1 — 容器名
#   $2 — 镜像名（可含标签，如 my_image:1.0）
#   $@ — 其余参数：会在默认参数之后透传给 docker run，可用于覆盖
# 退出码：
#   0  成功
#   1  参数错误
#   2  Docker 未安装或未运行
#   3  启动 / 创建失败
#

set -euo pipefail

########################################
# 打印用法
########################################
usage() {
    echo "Usage: $0 <container_name> <image[:tag]> [extra-docker-run-args]" >&2
    exit 1
}

########################################
# 检查 docker 服务
########################################
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

########################################
# 默认 docker run 参数
########################################
build_default_args() {
    # 当前工作目录，用于 -w
    local CUR_PWD="$PWD"

    # shell 数组比纯字符串安全
    DEFAULT_RUN_ARGS=(
        -it                       # 分配 tty 并保持 STDIN
        -d                        # 后台运行（与 -it 结合，相当于 -itd）
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

########################################
# 主流程
########################################
main() {
    if [[ $# -lt 1 ]]; then
        usage
    elif [[ $# -eq 1 ]]; then
        IMAGE=$1
        CONTAINER_NAME="auto_$(date +%s%N | cut -c1-16)"  # 16位时间戳
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

    # ---------- 判断容器状态 ----------
    if docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container '${CONTAINER_NAME}' is already running."
        exit 0
    elif docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Starting existing container '${CONTAINER_NAME}'..."
        if docker start "${CONTAINER_NAME}"; then
            echo "Container started successfully."
            exit 0
        else
            echo "ERROR: Failed to start container '${CONTAINER_NAME}'." >&2
            exit 3
        fi
    else
        echo "Creating and starting new container '${CONTAINER_NAME}' from image '${IMAGE}'..."
        if docker run "${DEFAULT_RUN_ARGS[@]}" "${EXTRA_ARGS[@]}" --name "${CONTAINER_NAME}" "${IMAGE}"; then
            echo "Container '${CONTAINER_NAME}' created and started successfully."
            exit 0
        else
            echo "ERROR: Failed to run new container." >&2
            exit 3
        fi
    fi
}

main "$@"
