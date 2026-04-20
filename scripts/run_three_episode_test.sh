#!/usr/bin/env bash
set -u

TARGET_EPISODES="${1:-3}"
MAX_SECONDS="${MAX_SECONDS:-420}"
RUN_NAME="${RUN_NAME:-codex_three_episode_test}"
LOG="${LOG:-/tmp/fear_three_episode_test_$(date +%Y%m%d_%H%M%S).log}"

cd /workspaces/clearpath_docker/clearpath_ws

setsid bash -lc "source /opt/ros/jazzy/setup.bash && \
source /workspaces/clearpath_docker/clearpath_ws/install/setup.bash && \
cd /workspaces/clearpath_docker/clearpath_ws && \
exec stdbuf -oL -eL ros2 launch fear_jackal_sim fear_training.launch.py \
manage_sim_process:=true \
reward_mode:=external_only \
evaluation_only:=false \
use_policy_network:=true \
fear_reactive_policy:=false \
fear_model_mode:=smann \
smann_fear_threshold:=0.50 \
run_name:=${RUN_NAME}" > "${LOG}" 2>&1 &

PID="$!"
echo "LAUNCH_PID=${PID}"
echo "LOG=${LOG}"

START="$(date +%s)"
LAST_COUNT="-1"

while true; do
  if [[ -f "${LOG}" ]]; then
    COUNT="$(grep -c 'Episode complete episode=' "${LOG}" || true)"
  else
    COUNT="0"
  fi

  if [[ "${COUNT}" != "${LAST_COUNT}" ]]; then
    echo "EPISODES_COMPLETED=${COUNT}"
    LAST_COUNT="${COUNT}"
  fi

  if [[ "${COUNT}" -ge "${TARGET_EPISODES}" ]]; then
    echo "TARGET_REACHED"
    break
  fi

  if ! kill -0 "${PID}" 2>/dev/null; then
    echo "LAUNCH_EXITED_BEFORE_TARGET"
    break
  fi

  NOW="$(date +%s)"
  if [[ "$((NOW - START))" -ge "${MAX_SECONDS}" ]]; then
    echo "TIMEOUT_BEFORE_TARGET"
    break
  fi

  sleep 2
done

if kill -0 "${PID}" 2>/dev/null; then
  kill -INT -"${PID}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "${PID}" 2>/dev/null; then
      break
    fi
    sleep 0.5
  done
  if kill -0 "${PID}" 2>/dev/null; then
    kill -TERM -"${PID}" 2>/dev/null || true
  fi
  wait "${PID}" 2>/dev/null || true
fi

echo "=== KEY LOG LINES ==="
grep -E \
  'Fear trainer scaffold|Reset simulator|Requested simulator reset|Retrying simulator reset|Waiting for fresh|Fresh post-reset|Started episode|Agent memory updated step=0|Episode complete|Post-reset observation|terminal collision|goal_reached=True|Managed simulation process|Traceback|RuntimeError|ERROR|FATAL' \
  "${LOG}" | tail -n 260 || true
echo "=== END KEY LOG LINES ==="
