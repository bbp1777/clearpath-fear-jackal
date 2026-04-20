#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-clearpath-jazzy-sim}"
FEAR_REPO_HOST_PATH="${FEAR_REPO_HOST_PATH:-/home/sting/Behavior-Intrinsic-Fear-main}"
FORCE_BUILD=0

for arg in "$@"; do
  case "${arg}" in
    --build|--rebuild)
      FORCE_BUILD=1
      ;;
    -h|--help)
      echo "Usage: ./scripts/dev_shell.sh [--build]"
      echo "  --build, --rebuild  Rebuild the Docker image before opening the shell."
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      echo "Usage: ./scripts/dev_shell.sh [--build]" >&2
      exit 1
      ;;
  esac
done

if command -v docker >/dev/null 2>&1; then
  DOCKER_BIN="docker"
elif command -v docker.exe >/dev/null 2>&1; then
  DOCKER_BIN="docker.exe"
else
  echo "Docker was not found in this shell." >&2
  echo "If you are using Docker Desktop on Windows, start Docker Desktop and enable WSL integration for Ubuntu-24.04." >&2
  exit 1
fi

if ! "${DOCKER_BIN}" version >/dev/null 2>&1; then
  echo "Docker is installed, but the daemon is not reachable." >&2
  echo "Start Docker Desktop on Windows, wait for it to finish starting, then retry." >&2
  exit 1
fi

if [ "${FORCE_BUILD}" -eq 1 ] || ! "${DOCKER_BIN}" image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  "${DOCKER_BIN}" build -t "${IMAGE_NAME}" .
else
  echo "Using existing Docker image ${IMAGE_NAME}. Pass --build to rebuild it."
fi

DOCKER_RUN_ARGS=(
  --rm
  -it
  -p 6006:6006
  -e DISPLAY="${DISPLAY:-}"
  -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}"
  -e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}"
  -e PULSE_SERVER="${PULSE_SERVER:-}"
  -e QT_X11_NO_MITSHM=1
  -v /mnt/wslg:/mnt/wslg
  -v /tmp/.X11-unix:/tmp/.X11-unix
  -v "$(pwd)":/workspaces/clearpath_docker
  -w /workspaces/clearpath_docker
)

if [ -d "${FEAR_REPO_HOST_PATH}" ]; then
  echo "Mounting Behavior-Intrinsic-Fear repo from ${FEAR_REPO_HOST_PATH}."
  DOCKER_RUN_ARGS+=( -v "${FEAR_REPO_HOST_PATH}":/workspaces/Behavior-Intrinsic-Fear-main )
else
  echo "Behavior-Intrinsic-Fear repo was not found at ${FEAR_REPO_HOST_PATH}." >&2
  echo "SMANN loading will stay disabled until that repo is available or FEAR_REPO_HOST_PATH is set." >&2
fi

"${DOCKER_BIN}" run "${DOCKER_RUN_ARGS[@]}" "${IMAGE_NAME}" bash
