"""Potential diagnostics must survive early death and repeated W&B step values."""
import json
from types import SimpleNamespace

import numpy as np

from empowerment_warp.utils.evaluation_metrics import save_potential_policy_probe
from empowerment_warp.tracking import WandbMirror
from empowerment_warp.training.sac.progress import progress_wandb_row


def test_probe_counts_reset_terminal_action_and_excludes_postmortem_positions(tmp_path):
    track = np.array([[[1., 0.], [11., 0.]], [[2., 0.], [999., 0.]], [[3., 0.], [999., 0.]]])
    initial = np.array([[0., 0.], [10., 0.]])
    alive = np.array([[True, True], [True, False], [True, False]])
    result = save_potential_policy_probe(tmp_path / 'probe.json', {'agent_xy': track},
                                        {'agent_xy': initial}, alive, {'agent_xy': 1.},
                                        checkpoint_step=123, seed=10)
    assert result['metrics']['agent_xy/cells_visited'] == 6  # 0,1,2,3,10,11
    assert result['metrics']['agent_xy/displacement_mean'] == 2  # (3 + 1) / 2
    assert result['metrics']['early_termination_fraction'] == .5
    assert result['metrics']['valid_steps_mean'] == 2
    arrays = np.load(result['positions'])
    np.testing.assert_array_equal(arrays['agent_xy_displacement'], [3, 1])
    assert arrays['checkpoint_step'] == 123
    assert arrays['agent_xy'].shape == (4, 2, 2)


def test_progress_exports_probe_to_evaluation_namespace(tmp_path):
    path = tmp_path / 'evaluation.json'
    path.write_text(json.dumps({'policy_mode': 'stochastic', 'summary': {},
                              'potential_policy': {'metrics': {'agent_xy/displacement_mean': 2.}}}))
    row = progress_wandb_row({'evaluation': str(path), 'checkpoint_step': 123,
                             'target_env_steps': 100})
    assert row['evaluation/potential_policy/agent_xy/displacement_mean'] == 2
    assert row['env_steps'] == 123


def test_locomotion_distances_share_endpoint_and_ignore_dead_lane_reset(tmp_path):
    agent = np.array([[[10., 10.], [13., 14.]], [[13., 10.], [999., 999.]]])
    ball = agent + np.array([0., 4.])
    result = save_potential_policy_probe(
        tmp_path / 'probe.json', {'agent_xy': agent, 'ball_xy': ball},
        {'agent_xy': agent[0], 'ball_xy': ball[0]},
        np.array([[True, True], [True, False]]),
        {'agent_xy': 1., 'ball_xy': 1.}, checkpoint_step=0, seed=0,
    )
    assert result['metrics']['agent_to_center_xy/distance_mean'] == 4.
    assert result['metrics']['agent_to_ball_xy/distance_mean'] == 4.


def test_block_distances_use_each_lanes_last_valid_endpoint(tmp_path):
    # Lane 1 ends after action one; its subsequent positions must be ignored.
    alive = np.array([[True, True], [True, False]])
    gripper = np.array([[[0., 0., 0.], [10., 0., 0.]],
                        [[1., 0., 0.], [999., 0., 0.]]])
    tracks = {'gripper_xyz': gripper}
    for i in range(3):
        tracks[f'block_{i}_xyz'] = gripper + np.array([i + 1., 0., 2.])
    tracks['block_0_xyz'][0, 1, 0] = 13.
    tracks['block_1_xyz'][0, 1, 0] = 11.
    # A postmortem block appears deceptively close.
    tracks['block_0_xyz'][1, 1] = gripper[1, 1]
    result = save_potential_policy_probe(
        tmp_path / 'probe.json', tracks, {k: v[0].copy() for k, v in tracks.items()},
        alive, {k: .05 for k in tracks}, checkpoint_step=123, seed=0,
    )
    assert result['metrics']['gripper_to_nearest_block_xy/distance_mean'] == 1.
    np.testing.assert_allclose(result['metrics']['gripper_to_nearest_block_xyz/distance_mean'], np.sqrt(5))
    with np.load(result['positions']) as arrays:
        np.testing.assert_allclose(arrays['gripper_to_blocks_xy_distance'], [[1, 2, 3], [3, 1, 3]])
    path = tmp_path / 'evaluation.json'
    path.write_text(json.dumps({'summary': {}, 'potential_policy': result}))
    row = progress_wandb_row({'evaluation': str(path), 'checkpoint_step': 123,
                             'target_env_steps': 100})
    assert row['evaluation/potential_policy/gripper_to_nearest_block_xy/distance_mean'] == 1.


def test_wandb_preserves_multiple_rows_at_same_environment_step(tmp_path):
    calls = []
    mirror = WandbMirror(tmp_path, run=SimpleNamespace(log=lambda *a, **kw: calls.append((a, kw))),
                         metadata={'id': 'test', 'url': 'local', '_step_key': 'env_steps'})
    mirror.log({'env_steps': 100, 'training/loss': 1})
    mirror.log({'env_steps': 100, 'evaluation/potential_policy/player/cells_visited': 5})
    assert len(calls) == 2
    assert all(kwargs == {} for _, kwargs in calls)  # no repeated explicit W&B step
    assert all(args[0]['env_steps'] == 100 for args, _ in calls)
    assert mirror.run is not None
