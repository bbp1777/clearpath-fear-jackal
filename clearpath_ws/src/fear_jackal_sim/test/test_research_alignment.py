import json
import numpy as np

from fear_jackal_sim.agent import FearAgent
from fear_jackal_sim.dataset_tools import (
    DANGER_CLASS_NUMBER,
    DANGER_REWARD_LABEL,
    SAFE_CLASS_NUMBER,
    SAFE_REWARD_LABEL,
    class_numbers_from_reward_labels,
    load_exported_smann_dataset,
    reward_labels_from_class_numbers,
)
from fear_jackal_sim.rl_types import AgentConfig, EnvironmentConfig, ObservationBundle
from fear_jackal_sim.smann import SMANNAdapter
from fear_jackal_sim.vision_utils import compute_green_goal_coverage


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


class _Environment:
    config = EnvironmentConfig(max_episode_steps=400)


def test_sparse_external_reward_only_goal_pays():
    agent = FearAgent(
        environment=_Environment(),
        config=AgentConfig(reward_mode='external_only', use_policy_network=False),
        logger=_Logger(),
    )

    assert agent.reward(ObservationBundle(goal_reached=False, collision=False)) == 0.0
    assert agent.reward(ObservationBundle(goal_reached=False, collision=True, terminal=True)) == 0.0
    assert agent.reward(ObservationBundle(goal_reached=True, goal_coverage=0.31)) == 1.0


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

    assert compute_green_goal_coverage(grass_like) == 0.0
    assert compute_green_goal_coverage(goal_block) == 1.0
