"""Spatial distances and step-zero evaluations survive every output layer."""
from types import SimpleNamespace
import json

import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.training.baselines.recipe import Recipe
from empowerment_warp.training.baselines import runner
from empowerment_warp.agents.ppo import empty_observation_norm
from empowerment_warp.media import evaluation as evaluation_media


@pytest.fixture(autouse=True)
def mock_rendering(monkeypatch):
    def render(source, output):
        output.mkdir(parents=True, exist_ok=True)
        video = output/'potential.mp4'
        video.write_bytes(b'unit-test rendering stub')
        return {'video':str(video), 'fps':24, 'metadata':str(output/'provenance.json')}
    monkeypatch.setattr(evaluation_media, 'render_evaluation', render)


def evaluation_case(environment, run_dir):
    recipe = Recipe(environment, 'gce_vector', 30, 1000, str(run_dir), num_envs=2,
                    rollout_steps=2, episode_steps=100, max_updates=1,
                    reset_preset='past_center_ball' if environment == 'ant' else None)
    obs = np.zeros((100, 128, recipe.model_environment.observation_dim), dtype=np.float32)
    obs[..., :2] = [14., 13.]
    if environment == 'ant_soccer':
        obs[..., 15:17] = [20., 13.]
    if environment == 'builderbench_corner3':
        obs[..., 10:19] = [17., 13., 0., 21., 13., 0., 24., 13., 0.]
    done = np.zeros((100, 128), dtype=bool)
    done[20, 0] = True
    done[99, 1] = True
    alive = np.ones((100, 128), dtype=bool)
    alive[21:, 0] = False
    if environment == 'builderbench_corner3':
        tracks = {'gripper_xyz':obs[..., :3].copy(),
                  **{f'block_{i}_xyz':obs[..., 10+3*i:13+3*i].copy() for i in range(3)}}
    else:
        tracks = {'agent_xy':obs[..., :2].copy()}
        if environment == 'ant_soccer':
            tracks['ball_xy'] = obs[..., 15:17].copy()
    initial = {f'initial_{name}':track[0].copy() for name,track in tracks.items()}
    # Only the terminal action moves lane zero. Later masked positions are far away.
    # Moving all entities together preserves each requested endpoint distance.
    for track in tracks.values():
        track[20, 0, :2] += [-1., 1.]
        track[21:, 0] = 999.
    arrays = dict(observations=obs, reached_observations=obs.copy(),
                  actions=np.zeros((100, 128, recipe.model_environment.action_dim)),
                  rewards=np.zeros((100, 128)), dones=done, alive=alive,
                  control_dt=np.asarray(.02), skills=np.zeros((128, 0)),
                  **tracks, **initial)
    for name, size in [('qpos', 22), ('qvel', 20), ('ctrl', 8)]:
        arrays[name] = np.zeros((100, size))
        arrays[f'initial_{name}'] = np.zeros((128, size))
    for name, size in [('mocap_pos', 3), ('mocap_quat', 4)]:
        arrays[name] = np.zeros((100, 1, size))
        arrays[f'initial_{name}'] = np.zeros((128, 1, size))
    functions = {'diagnostic':lambda estimator, moments:{'probe_all_finite':True},
                 'standard_evaluation':lambda params, reset_key, action_key, skills:arrays,
                 'environment_config':{'ctrl_dt':.02}}
    state = {'update':jnp.asarray(0), 'ppo':SimpleNamespace(params={}),
             'estimator':SimpleNamespace(params={}),
             'controller':SimpleNamespace(observation_norm=empty_observation_norm(recipe.model_environment.observation_dim))}
    return recipe, state, functions


@pytest.mark.parametrize('environment,metric,expected', [
    ('ant', 'agent_center_distance_xy_mean', 5.),
    ('ant_soccer', 'agent_ball_distance_xy_mean', 6.),
    ('builderbench_corner3', 'gripper_nearest_block_distance_xy_mean', 3.),
])
def test_actual_evaluate_keeps_spatial_distance_and_compact_npz(tmp_path, environment, metric, expected):
    recipe, state, functions = evaluation_case(environment, tmp_path)
    row = runner.evaluate(recipe, state, functions, tmp_path)
    assert row[f'stochastic/spatial/{metric}'] == expected
    assert row['stochastic/spatial/potential_endpoint_distance_xy_mean'] == expected
    assert row['stochastic/spatial/potential_endpoint_early_termination_count'] == 1
    assert row['evaluation_worlds'] == 128
    directory = tmp_path/'evaluations'/'step_000000000000'
    saved = json.loads((directory/'metrics.json').read_text())
    track_name = 'gripper_xyz' if environment == 'builderbench_corner3' else 'agent_xy'
    assert saved['potential_policy']['metrics'][f'{track_name}/displacement_mean'] == pytest.approx(np.sqrt(2.)/128)
    assert expected != saved['potential_policy']['metrics'][f'{track_name}/displacement_mean']
    with np.load(directory/'metrics_potential_positions.npz') as data:
        assert data['alive'].shape == (100, 128)
        assert data['alive'][:, 0].sum() == 21 and data['alive'][20, 0]
        positions = data[track_name]
        assert positions.shape[:2] == (101, 128)
        np.testing.assert_array_equal(positions[21, 0, :2], [13., 14.])
        assert 'observations' not in data.files
    with np.load(directory/'trajectories.npz') as data:
        metadata = json.loads(str(data['metadata']))
        assert metadata['checkpoint_step'] == metadata['target_env_steps'] == 0
        assert metadata['policy_mode'] == 'stochastic' and metadata['num_envs'] == 128
    assert 'spatial/' in (tmp_path/'eval.csv').read_text()


def test_run_mirrors_initial_evaluation_after_local_save(tmp_path, monkeypatch):
    recipe, state, functions = evaluation_case('ant', tmp_path/'run')
    logged = []

    class Mirror:
        def log(self, row):
            local = [json.loads(line) for line in (tmp_path/'run/metrics.jsonl').read_text().splitlines()]
            assert local[-1] == row
            if row['segment'] == 0:
                assert (tmp_path/'run/evaluations/step_000000000000/metrics_potential_positions.npz').exists()
            logged.append(dict(row))

        def log_video(self, row, **kwargs):
            directory = tmp_path/'run/evaluations'/f"step_{row['target_step']:012d}"
            assert (directory/'metrics.json').exists()
            assert (directory/'metrics_potential_positions.npz').exists()
            assert (directory/'trajectories.npz').exists()
            assert (tmp_path/'run/eval.csv').exists()
            return True

        def finish(self):
            pass

    def update(recipe, state, functions):
        state['update'] += 1
        return {'env_steps':int(state['update'])*recipe.steps_per_update}

    monkeypatch.setattr(runner.jax, 'devices', lambda:[SimpleNamespace(platform='gpu', device_kind='test device')])
    monkeypatch.setattr(runner, 'build', lambda recipe:(state, functions))
    monkeypatch.setattr(runner, 'update', update)
    monkeypatch.setattr(runner.WandbMirror, 'start', lambda *args, **kwargs:Mirror())
    monkeypatch.setattr(runner.CheckpointStore, 'save', lambda *args, **kwargs:None)
    runner.run(recipe)
    initial = logged[0]
    assert initial['segment'] == 0 and initial['env_steps'] == 0
    assert initial['stochastic/spatial/agent_center_distance_xy_mean'] == 5.
    assert logged[1]['segment'] == 1
    assert logged[1]['env_steps'] == recipe.steps_per_update
