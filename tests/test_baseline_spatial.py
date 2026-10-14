"""Spatial metric logging preserves displacement and handles terminal endpoints."""
import json
from pathlib import Path
import runpy

import numpy as np

module = runpy.run_path(str(Path(__file__).parents[1] / 'src/empowerment_warp/training/baselines/spatial.py'))


def test_ant_endpoint_excludes_resets_but_average_keeps_full_rollout():
    obs = np.zeros((110, 2, 20))
    reached = np.full_like(obs, 100.)
    reached[1, 0, :2] = [7, 6]
    reached[99, 1, :2] = [13, 14]
    dones = np.zeros((110, 2), bool)
    dones[1, 0] = True
    row, positions = module['spatial_metrics']('ant', obs, reached, dones)
    assert row['potential_endpoint_distance_xy_mean'] == 5
    assert row['agent_center_distance_xy_mean'] > 100
    assert row['potential_endpoint_early_termination_count'] == 1
    assert positions['agent_xy'].shape == (101, 2, 2)


def test_soccer_logs_moving_ball_endpoint(tmp_path):
    obs = np.zeros((110, 2, 20))
    after = obs.copy()
    after[..., 15:17] = [3, 4]
    row = module['save_spatial_evaluation'](tmp_path, 'ant_soccer', 'stochastic', 500, obs, after, np.zeros((110, 2)))
    assert row['stochastic/spatial/potential_endpoint_distance_xy_mean'] == 5
    with np.load(tmp_path/'spatial/stochastic_step_000000000500.npz') as z:
        assert z['ball_xy'].shape == (101, 2, 2)
        assert json.loads(str(z['metadata']))['horizon_actions'] == 100


def test_blocks_use_nearest_xy_and_do_not_modify_displacement_inputs():
    obs = np.zeros((100, 2, 51))
    after = obs.copy()
    after[..., 10:19] = [3, 4, 90, 0, 6, 0, 8, 0, 0]
    before = after.copy()
    row, positions = module['spatial_metrics']('builderbench_corner3', obs, after, np.zeros((100, 2)))
    assert row['potential_endpoint_distance_xy_mean'] == 5
    assert row['gripper_nearest_block_distance_xy_mean'] == 5
    assert positions['block_xyz'].shape == (101, 2, 3, 3)
    np.testing.assert_array_equal(after, before)
