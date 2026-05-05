# Clearpath Jackal Fear-RL Sidewalk Simulator

This repository contains the ROS 2 Jazzy / Gazebo Harmonic Jackal simulation used for sparse-reward PPO and PPO + SMANN fear experiments in the sidewalk environment. The Sanchez `Behavior-Intrinsic-Fear` codebase is vendored directly into this repository, so no external clone is required after download.

The main experiment compares:

- **Base PPO**: sparse external reward only
- **PPO + SMANN fear**: the same PPO setup with a frozen offline-trained SMANN model that adds thresholded negative intrinsic reward for unsafe behavior

The task reward is sparse: the agent receives `+1` only when the RGB image reaches the configured goal-coverage threshold. Grass and obstacle contacts terminate the episode but do not add an external penalty. The default training setup uses 7 seeds, 50 episodes per seed, 750 control steps per episode, and a 0.5 s control period.

## Repository Layout

- `Behavior-Intrinsic-Fear-main/`: vendored upstream Sanchez code used by the SMANN adapter at runtime
- `Dockerfile`: ROS 2 Jazzy / Clearpath simulation container
- `scripts/`: development shell and smoke-test helpers
- `sim_setup/`: Clearpath robot configuration and generated launch/config files used by the simulator
- `clearpath_ws/src/fear_jackal_sim/`: custom ROS 2 package with the trainer, goal monitor, SMANN adapter, dataset tooling, launch files, and tests
- `clearpath_ws/src/jackal_smann_eval/`: optional frozen evaluation package for one-shot SMANN runs
- `clearpath_ws/logs/manual_dataset/`: tracked 63-sample Jackal RGB-D low-shot dataset for offline SMANN training

Generated build products, logs, checkpoints, TensorBoard runs, episode archives, and paper figures are intentionally ignored by git.

## Manual Dataset

The checked-in dataset under `clearpath_ws/logs/manual_dataset/` was created for this Jackal setup and is not part of the upstream Sanchez repository.

Label convention:

- unsafe behavior: reward label `-1`, classifier class `0`
- safe behavior: reward label `0`, classifier class `1`

The dataset uses the Jackal-prefixed three-step filenames:

- `Jackal-v0_lookback_3observations.npy`
- `Jackal-v0_lookback_3class.npy`
- `Jackal-v0_lookback_3class_number.npy`
- `Jackal-v0_lookback_3reward.npy`

Additional dataset provenance is stored in `clearpath_ws/src/fear_jackal_sim/config/manual_dataset_metadata.json`.

## Build

From WSL:

```bash
cd /home/sting/clearpath_docker
chmod +x scripts/dev_shell.sh
./scripts/dev_shell.sh --build
```

Inside the container:

```bash
source /opt/ros/jazzy/setup.bash
cd /workspaces/clearpath_docker/clearpath_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select fear_jackal_sim jackal_smann_eval
source install/setup.bash
```

## Offline SMANN Training

Run the 5-fold grid search and train the selected final checkpoint:

```bash
ros2 run fear_jackal_sim train_smann_grid \
  --dataset-dir /workspaces/clearpath_docker/clearpath_ws/logs/manual_dataset \
  --fear-repo-path /workspaces/clearpath_docker/Behavior-Intrinsic-Fear-main/CarRacingTesting \
  --output-dir /workspaces/clearpath_docker/clearpath_ws/logs/smann_training/smann_grid
```

Train a single SMANN checkpoint directly:

```bash
ros2 run fear_jackal_sim train_smann \
  --dataset-dir /workspaces/clearpath_docker/clearpath_ws/logs/manual_dataset \
  --checkpoint-dir /workspaces/clearpath_docker/clearpath_ws/logs/smann_training/manual_selected/weights \
  --fear-repo-path /workspaces/clearpath_docker/Behavior-Intrinsic-Fear-main/CarRacingTesting \
  --run-name smann_offline_manual_63
```

Each offline SMANN run writes loss curves and metrics next to the checkpoint directory.

## Training Runs

Set a run root and the final SMANN checkpoint you want to evaluate:

```bash
export RUN_ROOT=/workspaces/clearpath_docker/clearpath_ws/logs/runs_rgbdcnn
export FINAL_SMANN=/workspaces/clearpath_docker/clearpath_ws/logs/smann_training/smann_grid/final_selected/weights
```

### Base PPO

Single-seed example:

```bash
ros2 launch fear_jackal_sim fear_training.launch.py \
  manage_sim_process:=true \
  reward_mode:=external_only \
  evaluation_only:=false \
  use_policy_network:=true \
  fear_model_mode:=none \
  max_episode_steps:=750 \
  control_period_s:=0.5 \
  goal_completion_threshold:=0.50 \
  max_episodes:=50 \
  random_seed:=1 \
  run_artifact_dir:="$RUN_ROOT" \
  run_name:=ppo_rgbdcnn_base_seed01
```

Seven-seed sweep:

```bash
for SEED_PAD in 01 02 03 04 05 06 07; do
  SEED=$((10#$SEED_PAD))

  ros2 launch fear_jackal_sim fear_training.launch.py \
    manage_sim_process:=true \
    reward_mode:=external_only \
    evaluation_only:=false \
    use_policy_network:=true \
    fear_model_mode:=none \
    max_episode_steps:=750 \
    control_period_s:=0.5 \
    goal_completion_threshold:=0.50 \
    max_episodes:=50 \
    random_seed:=$SEED \
    run_artifact_dir:="$RUN_ROOT" \
    run_name:=ppo_rgbdcnn_base_seed${SEED_PAD}
done
```

### PPO + SMANN Threshold Sweep

```bash
for ITEM in "0.25 t025" "0.50 t050" "0.75 t075"; do
  set -- $ITEM
  THRESH=$1
  TAG=$2

  for SEED_PAD in 01 02 03 04 05 06 07; do
    SEED=$((10#$SEED_PAD))

    ros2 launch fear_jackal_sim fear_training.launch.py \
      manage_sim_process:=true \
      reward_mode:=combined \
      evaluation_only:=false \
      use_policy_network:=true \
      fear_model_mode:=smann \
      smann_checkpoint:="$FINAL_SMANN" \
      smann_dataset_dir:=/workspaces/clearpath_docker/clearpath_ws/logs/manual_dataset \
      smann_fear_threshold:=$THRESH \
      enable_online_smann_updates:=false \
      intrinsic_reward_scale:=1.0 \
      max_episode_steps:=750 \
      control_period_s:=0.5 \
      goal_completion_threshold:=0.50 \
      max_episodes:=50 \
      random_seed:=$SEED \
      run_artifact_dir:="$RUN_ROOT" \
      run_name:=ppo_rgbdcnn_smann_${TAG}_seed${SEED_PAD}
  done
done
```

SMANN intrinsic reward is thresholded in the Sanchez style:

```text
P(unsafe) < threshold  -> intrinsic reward 0
P(unsafe) >= threshold -> intrinsic reward -P(unsafe)
```

## TensorBoard

TensorBoard logs are written to:

```text
/workspaces/clearpath_docker/clearpath_ws/logs/tensorboard
```

Start TensorBoard inside the container:

```bash
tensorboard --logdir /workspaces/clearpath_docker/clearpath_ws/logs/tensorboard --bind_all
```

## Tests

```bash
cd /workspaces/clearpath_docker/clearpath_ws/src/fear_jackal_sim
source /opt/ros/jazzy/setup.bash
PYTHONPATH=$PWD:$PYTHONPATH pytest -q test/test_research_alignment.py
```

## Notes

- The vendored `Behavior-Intrinsic-Fear-main` directory is the runtime source of truth for the SMANN adapter.
- Generated run outputs under `clearpath_ws/logs/` are local working artifacts and are not versioned.
- The tracked manual dataset is the only `logs/` content intentionally kept in git.
