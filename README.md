# Clearpath Jackal Fear-RL Sidewalk Simulator

This repository contains the ROS 2 Jazzy / Gazebo Harmonic Jackal simulation used for a sparse-reward fear-intrinsic reinforcement learning experiment.

The main experiment compares:

- **Base PPO**: sparse external reward only.
- **PPO + Sanchez SMANN fear**: the same PPO setup with a frozen offline-trained SMANN model that adds thresholded negative intrinsic reward for unsafe behavior.

The goal reward is sparse: the agent receives `+1` only when the RGB image reaches the configured green goal-block coverage threshold. Grass and obstacle contacts terminate the episode but do not add an external penalty.

## Repository Layout

- `Dockerfile`: ROS 2 Jazzy / Clearpath simulation container.
- `scripts/`: development and smoke-test helpers.
- `sim_setup/`: Clearpath robot configuration and generated launch/config files used by the simulator.
- `clearpath_ws/src/fear_jackal_sim/`: custom ROS 2 package with the trainer, Gazebo world, goal monitor, SMANN adapter, TensorBoard logging, and tests.
- `clearpath_ws/logs/rodney_dataset/`: manually created 63-sample Jackal RGB-D low-shot dataset for offline SMANN training.

Generated build products, TensorBoard runs, episode archives, and trained weights are intentionally ignored by git. Trained SMANN weights should be regenerated from the manual dataset or shared separately as release artifacts.

## Manual Dataset

The checked-in dataset under `clearpath_ws/logs/rodney_dataset/` is not from the upstream Sanchez repository. It was manually created for this Jackal setup.

Label convention:

- unsafe behavior: reward label `-1`, classifier class `0`
- safe behavior: reward label `0`, classifier class `1`

The dataset uses the Rodney-style prefix filenames:

- `Jackal-v0_lookback_3observations.npy`
- `Jackal-v0_lookback_3class.npy`
- `Jackal-v0_lookback_3class_number.npy`
- `Jackal-v0_lookback_3reward.npy`

The project loader supports both these prefixed filenames and canonical `observations.npy` / `class.npy` / `class_number.npy` names.

Additional dataset provenance is stored in `clearpath_ws/src/fear_jackal_sim/config/manual_dataset_metadata.json`.

## Build

From WSL:

```bash
cd /home/sting/clearpath_docker
chmod +x scripts/dev_shell.sh
./scripts/dev_shell.sh
```

Inside the container:

```bash
source /opt/ros/jazzy/setup.bash
cd /workspaces/clearpath_docker/clearpath_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select fear_jackal_sim
source install/setup.bash
```

## Run Base PPO

```bash
ros2 launch fear_jackal_sim fear_training.launch.py \
  manage_sim_process:=true \
  reward_mode:=external_only \
  evaluation_only:=false \
  use_policy_network:=true \
  fear_model_mode:=none \
  run_name:=ppo_sparse_base_001
```

In `external_only`, the trainer disables fear scoring so the baseline stays clean.

## Run PPO + SMANN Fear

First train or regenerate the SMANN checkpoint from the manual dataset:

```bash
ros2 run fear_jackal_sim train_smann \
  --dataset-dir /workspaces/clearpath_docker/clearpath_ws/logs/rodney_dataset \
  --checkpoint-dir /workspaces/clearpath_docker/clearpath_ws/logs/rodney_training/jackal_mann_independent/weights \
  --fear-repo-path /workspaces/Behavior-Intrinsic-Fear-main/CarRacingTesting \
  --run-name smann_offline_manual_63
```

Then run PPO with frozen SMANN intrinsic fear:

```bash
ros2 launch fear_jackal_sim fear_training.launch.py \
  manage_sim_process:=true \
  reward_mode:=combined \
  evaluation_only:=false \
  use_policy_network:=true \
  fear_model_mode:=smann \
  smann_fear_threshold:=0.50 \
  enable_online_smann_updates:=false \
  run_name:=ppo_sparse_smann_t050_001
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

Important scalar groups include:

- `episode/external_return`
- `episode/intrinsic_return`
- `episode/combined_return`
- `episode/goal_reached`
- `episode/terminal_collision`
- `episode/goal_coverage_final`
- `fear/raw_unsafe_probability_mean`
- `fear/raw_unsafe_probability_max`
- `fear/active_fraction`
- `ppo/actor_loss`
- `ppo/critic_loss`
- `ppo/entropy`
- `ppo/clip_fraction`

## Source Audit

To compare the local Sanchez code against the canonical upstream repository:

```bash
ros2 run fear_jackal_sim sanchez_source_audit
```

The audit writes reports under:

```text
/workspaces/clearpath_docker/clearpath_ws/logs/source_audits
```

This is useful for paper provenance, but it is not required for the simulator or training loop to run.

## Archive Generated Outputs Before Fresh Runs

Dry run:

```bash
ros2 run fear_jackal_sim fear_archive_outputs --dry-run
```

Archive generated outputs while preserving the manual dataset:

```bash
ros2 run fear_jackal_sim fear_archive_outputs
```

## Tests

```bash
cd /workspaces/clearpath_docker/clearpath_ws/src/fear_jackal_sim
source /opt/ros/jazzy/setup.bash
PYTHONPATH=$PWD:$PYTHONPATH pytest -q test/test_research_alignment.py
```
