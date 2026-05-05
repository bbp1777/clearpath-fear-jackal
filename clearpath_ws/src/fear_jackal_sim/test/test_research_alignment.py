import json
import numpy as np
import pytest
from sensor_msgs.msg import Image

from fear_jackal_sim.agent import (
    ActorNetwork,
    CriticNetwork,
    FearAgent,
    POLICY_IMAGE_SIZE,
    POLICY_INPUT_CHANNELS,
    torch,
)
from fear_jackal_sim.dataset_tools import (
    DANGER_CLASS_NUMBER,
    DANGER_REWARD_LABEL,
    SAFE_CLASS_NUMBER,
    SAFE_REWARD_LABEL,
    class_numbers_from_reward_labels,
    load_exported_smann_dataset,
    reward_labels_from_class_numbers,
)
from fear_jackal_sim.rl_types import AgentConfig, EnvironmentConfig, ObservationBundle, TrainerConfig
from fear_jackal_sim.smann import SMANNAdapter
from fear_jackal_sim.vision_utils import compute_green_goal_coverage


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


class _Environment:
    config = EnvironmentConfig(max_episode_steps=500)


def _rgb_image_msg(width: int = POLICY_IMAGE_SIZE, height: int = POLICY_IMAGE_SIZE) -> Image:
    msg = Image()
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 1] = 255
    msg.height = height
    msg.width = width
    msg.encoding = 'rgb8'
    msg.step = width * 3
    msg.data = rgb.tobytes()
    return msg


def _depth_image_msg(width: int = POLICY_IMAGE_SIZE, height: int = POLICY_IMAGE_SIZE) -> Image:
    msg = Image()
    depth = np.full((height, width), 1.5, dtype=np.float32)
    msg.height = height
    msg.width = width
    msg.encoding = '32FC1'
    msg.step = width * np.dtype(np.float32).itemsize
    msg.data = depth.tobytes()
    return msg


def test_final_paper_episode_defaults():
    environment_config = EnvironmentConfig()
    trainer_config = TrainerConfig()

    assert environment_config.max_episode_steps == 500
    assert environment_config.goal_completion_threshold == 0.50
    assert trainer_config.control_period_s == 0.5
    assert trainer_config.max_episodes == 50
    assert trainer_config.control_period_s * environment_config.max_episode_steps == 250.0


def test_sparse_external_reward_only_goal_pays():
    agent = FearAgent(
        environment=_Environment(),
        config=AgentConfig(reward_mode='external_only', use_policy_network=False),
        logger=_Logger(),
    )

    assert agent.reward(ObservationBundle(goal_reached=False, collision=False)) == 0.0
    assert agent.reward(ObservationBundle(goal_reached=False, collision=True, terminal=True)) == 0.0
    assert agent.reward(ObservationBundle(goal_reached=True, goal_coverage=0.50)) == 1.0


def test_policy_observation_uses_single_rgbd_frame():
    if torch is None:
        pytest.skip('PyTorch is unavailable in this environment.')

    agent = FearAgent(
        environment=_Environment(),
        config=AgentConfig(use_policy_network=False),
        logger=_Logger(),
    )
    observation = ObservationBundle(
        color_msg=_rgb_image_msg(),
        depth_msg=_depth_image_msg(),
    )

    policy_tensor = agent._policy_observation(observation)

    assert tuple(policy_tensor.shape) == (POLICY_INPUT_CHANNELS, POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE)
    assert policy_tensor.dtype == torch.float32


def test_rodney_style_actor_critic_accept_rgbd_frames():
    if torch is None or ActorNetwork is None or CriticNetwork is None:
        pytest.skip('PyTorch is unavailable in this environment.')

    actor = ActorNetwork(action_dim=6)
    critic = CriticNetwork()
    batch = torch.zeros((2, POLICY_INPUT_CHANNELS, POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE), dtype=torch.float32)

    action_probs = actor(batch)
    state_values = critic(batch)

    assert tuple(action_probs.shape) == (2, 6)
    assert tuple(state_values.shape) == (2, 1)
    assert torch.all(action_probs >= 0.0)
    assert torch.allclose(action_probs.sum(dim=-1), torch.ones(2), atol=1.0e-5)


def test_none_fear_mode_yields_zero_intrinsic_reward():
    agent = FearAgent(
        environment=_Environment(),
        config=AgentConfig(reward_mode='external_only', fear_model_mode='none', use_policy_network=False),
        logger=_Logger(),
    )

    assert agent.fear_model_mode == 'none'
    assert agent._compute_intrinsic_reward(None) == 0.0


def test_reward_labels_match_paper_convention():
    class_numbers = np.asarray([DANGER_CLASS_NUMBER, SAFE_CLASS_NUMBER], dtype=np.int64)
    reward_labels = reward_labels_from_class_numbers(class_numbers)

    assert reward_labels.tolist() == [DANGER_REWARD_LABEL, SAFE_REWARD_LABEL]
    assert class_numbers_from_reward_labels(reward_labels).tolist() == [
        DANGER_CLASS_NUMBER,
        SAFE_CLASS_NUMBER,
    ]


def test_exported_dataset_loads_reward_label_metadata(tmp_path):
    observations = np.zeros((2, 3, 4, 84, 84), dtype=np.uint8)
    class_names = np.asarray(['danger', 'safe'])
    class_numbers = np.asarray([DANGER_CLASS_NUMBER, SAFE_CLASS_NUMBER], dtype=np.int64)
    np.save(tmp_path / 'observations.npy', observations)
    np.save(tmp_path / 'class.npy', class_names)
    np.save(tmp_path / 'class_number.npy', class_numbers)
    with open(tmp_path / 'metadata.json', 'w', encoding='ascii') as handle:
        json.dump({'source': 'manual_jackal_low_shot'}, handle)

    _, _, _, metadata = load_exported_smann_dataset(str(tmp_path))

    assert metadata['unsafe_reward_label_count'] == 1
    assert metadata['safe_reward_label_count'] == 1
    assert metadata['reward_label_map']['danger'] == DANGER_REWARD_LABEL


def test_exported_dataset_loader_accepts_rodney_prefixed_files(tmp_path):
    prefix = tmp_path / 'Jackal-v0_lookback_3'
    observations = np.zeros((2, 3, 4, 84, 84), dtype=np.uint8)
    class_names = np.asarray(['danger', 'safe'])
    class_numbers = np.asarray([DANGER_CLASS_NUMBER, SAFE_CLASS_NUMBER], dtype=np.int64)
    rewards = np.asarray([DANGER_REWARD_LABEL, SAFE_REWARD_LABEL], dtype=np.int64)
    np.save(str(prefix) + 'observations.npy', observations)
    np.save(str(prefix) + 'class.npy', class_names)
    np.save(str(prefix) + 'class_number.npy', class_numbers)
    np.save(str(prefix) + 'reward.npy', rewards)

    loaded_observations, _, loaded_classes, metadata = load_exported_smann_dataset(str(tmp_path))

    assert loaded_observations.shape == observations.shape
    assert loaded_classes.tolist() == class_numbers.tolist()
    assert metadata['unsafe_reward_label_count'] == 1
    assert metadata['safe_reward_label_count'] == 1


def test_smann_thresholded_intrinsic_reward():
    adapter = SMANNAdapter(fear_threshold=0.5)

    def below_threshold(_window, _logger):
        adapter.last_raw_score = 0.49
        return 0.49

    adapter.predict_unsafe_probability = below_threshold
    assert adapter.compute_thresholded_intrinsic_reward(np.zeros((3, 4, 84, 84)), _Logger()) == 0.0
    assert adapter.last_fear_active is False

    def above_threshold(_window, _logger):
        adapter.last_raw_score = 0.75
        return 0.75

    adapter.predict_unsafe_probability = above_threshold
    assert adapter.compute_thresholded_intrinsic_reward(np.zeros((3, 4, 84, 84)), _Logger()) == -0.75
    assert adapter.last_fear_active is True


def test_rgb_goal_mask_does_not_count_grass():
    grass_like = np.zeros((16, 16, 3), dtype=np.uint8)
    grass_like[:, :] = [45, 150, 55]
    goal_block = np.zeros((16, 16, 3), dtype=np.uint8)
    goal_block[:, :] = [0, 255, 0]
    dim_gazebo_goal_block = np.zeros((16, 16, 3), dtype=np.uint8)
    dim_gazebo_goal_block[:, :] = [0, 150, 0]

    assert compute_green_goal_coverage(grass_like) == 0.0
    assert compute_green_goal_coverage(goal_block) == 1.0
    assert compute_green_goal_coverage(dim_gazebo_goal_block) == 1.0
