"""Baseline evaluation uses saved moments, shared world keys, and charged budgets."""
from dataclasses import replace
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.training.baselines import runner
from empowerment_warp.agents.ppo import empty_observation_norm, normalize_observations
from empowerment_warp.training.baselines.recipe import ControllerPPOConfig, Recipe
from empowerment_warp.evaluation_protocol import evaluation_keys
from empowerment_warp.recipes import resolve


def recipe(environment='ant', **overrides):
    return Recipe(environment, 'gce_vector', 30, 10_000, 'unused',
                  num_envs=4, rollout_steps=4, reset_preset='past_center_ball' if environment == 'ant' else None,
                  **overrides)


class Data(NamedTuple):
    qpos: object
    qvel: object
    ctrl: object
    mocap_pos: object
    mocap_quat: object


class State(NamedTuple):
    data: object
    obs: object
    terminated: object
    truncated: object
    reward: object
    clock: object


class TinyEvaluationEnvironment:
    action_size = 8
    soccer = False
    config = SimpleNamespace(num_envs=128, ctrl_dt=.1)

    def reset(self, keys):
        obs = jax.vmap(lambda key: jax.random.uniform(key, (42,), minval=2., maxval=4.))(keys)
        zeros = jnp.zeros(len(keys), bool)
        return State(Data(*(obs,)*5), obs, zeros, zeros, jnp.zeros(len(keys)),
                     jnp.zeros(len(keys), jnp.int32))

    def step(self, state, actions):
        obs = state.obs.at[:, :8].add(actions*.1)
        clock = state.clock+1
        # One lane ends early; another terminates on the last requested action.
        terminated = ((jnp.arange(128) == 0) & (clock == 3)
                      | (jnp.arange(128) == 1) & (clock == 100))
        return state._replace(data=Data(*(obs,)*5), obs=obs, clock=clock,
                              terminated=terminated, reward=jnp.ones(128))

    def where_done(self, state, reset):
        done = state.terminated | state.truncated
        return jax.tree.map(lambda a, b: jnp.where(
            done.reshape(done.shape+(1,)*(a.ndim-1)), b, a), state, reset)


class TinyPolicy:
    def apply(self, variables, observations):
        mean = observations[:, :8]*variables['params']['scale']
        return SimpleNamespace(mean=lambda: mean, stddev=lambda: jnp.ones_like(mean)*.3), None


def test_adapter_uses_saved_moments_per_world_keys_and_first_terminal_action():
    cfg = recipe(ppo=ControllerPPOConfig(normalize_actor_observations=True))
    env = TinyEvaluationEnvironment()
    norm = empty_observation_norm(42)._replace(mean=jnp.ones(42), variance=jnp.ones(42)*4)
    params = {'scale': jnp.asarray(.2)}
    reset_key, action_key, _ = evaluation_keys('ant', 0, 0)
    evaluate = runner.make_standard_evaluator(cfg, TinyPolicy(), env)
    arrays = evaluate((params, norm), reset_key, action_key, jnp.zeros((128, 0)))
    assert arrays['alive'].shape == (100, 128)
    np.testing.assert_array_equal(arrays['alive'][:, 0], np.arange(100) < 3)
    assert np.asarray(arrays['alive'][:, 1]).all()
    assert arrays['dones'][2, 0] and arrays['dones'][99, 1]
    assert arrays['alive'][2, 0], 'the first terminal action contributes its reached position'
    _, step_key = jax.random.split(action_key)
    world_keys = jax.random.split(step_key, 128)
    observations = normalize_observations(norm, arrays['observations'][0])
    noise = jax.vmap(lambda key: jax.random.normal(key, (8,)))(world_keys)
    np.testing.assert_allclose(arrays['actions'][0], jnp.tanh(observations[:, :8]*.2+.3*noise), atol=1e-6)
    repeated = evaluate((params, norm), reset_key, action_key, jnp.zeros((128, 0)))
    np.testing.assert_array_equal(arrays['actions'], repeated['actions'])
    next_keys = evaluation_keys('ant', 0, 50_000_000)
    fresh = evaluate((params, norm), next_keys[0], next_keys[1], jnp.zeros((128, 0)))
    assert not np.array_equal(arrays['initial_qpos'], fresh['initial_qpos'])
    assert not np.array_equal(arrays['actions'], fresh['actions'])
    np.testing.assert_array_equal(norm.mean, jnp.ones(42))


def test_evaluation_environment_does_not_resize_training_recipe(monkeypatch):
    cfg = recipe()
    seen = []
    monkeypatch.setattr(runner, 'make_benchmark_environment',
                        lambda original, **kwargs: seen.append((original, kwargs)) or TinyEvaluationEnvironment())
    runner.make_standard_evaluator(cfg, TinyPolicy())
    assert seen == [(cfg, {'num_envs': 128})]
    assert cfg.num_envs == cfg.ppo.num_envs == 4


@pytest.mark.parametrize('environment,cadence', [
    ('ant', 50_000_000), ('ant_soccer', 50_000_000), ('builderbench_corner3', 5_000_000),
])
def test_resolved_cadence_and_evaluation_work(environment, cadence):
    cfg = recipe(environment)
    assert cfg.eval_every_env_steps == cadence
    scale = cfg.resolved()['effective_scale']
    assert scale['evaluation_worlds'] == 128 and scale['evaluation_horizon_actions'] == 100
    assert scale['evaluation_steps_per_event'] == 12_800+cfg.num_envs*cfg.horizon
    assert not scale['evaluation_steps_counted_in_training_budget']


def test_charge_crossings_floor_budget_and_smoke_final_target():
    cfg = recipe(eval_every_env_steps=1_000)
    assert runner.evaluation_targets(cfg, 999, 1_001) == [1_000]
    assert runner.evaluation_targets(cfg, 950, 2_100) == [1_000, 2_000]
    actual = cfg.updates*cfg.steps_per_update
    assert actual < cfg.total_env_steps
    assert runner.evaluation_targets(cfg, actual-cfg.steps_per_update, actual, final=True)[-1] == actual
    smoke = replace(cfg, max_updates=1)
    assert runner.evaluation_targets(smoke, 0, cfg.steps_per_update, final=True) == [cfg.steps_per_update]
    with pytest.raises(ValueError, match='positive'):
        replace(cfg, eval_every_env_steps=0)


def test_evaluate_saves_shared_payload_and_preserves_distance_csv_names(tmp_path, monkeypatch):
    cfg = recipe()
    norm = empty_observation_norm(42)
    state = dict(update=jnp.asarray(1), ppo=SimpleNamespace(params={'actor': jnp.asarray(2.)}),
                 controller=SimpleNamespace(observation_norm=norm),
                 estimator=SimpleNamespace(params={'channel': jnp.asarray(3.)}))
    arrays = {'alive': np.ones((100, 128), bool)}
    arrays['alive'][3:, 0] = False
    calls = []

    def rollout(parameters, reset_key, action_key, skills):
        assert parameters[0] is state['ppo'].params and parameters[1] is norm
        assert skills.shape == (128, 0)
        for actual, expected in zip((reset_key, action_key), evaluation_keys('ant', 0, 1000)[:2]):
            np.testing.assert_array_equal(actual, expected)
        return arrays

    def save(output, values, **kwargs):
        calls.append((output, values, kwargs))
        return dict(potential_policy={'metrics': {'agent_to_center_xy/distance_mean': 2.,
                    'horizon_steps': 100., 'num_envs': 128.}},
                    summary={'potential_agent_disp_xy': {'mean': .4}},
                    media={'video': str(output/'media'/'evaluation.mp4'), 'fps': 10})

    monkeypatch.setattr(runner, 'save_evaluation', save)
    functions = dict(standard_evaluation=rollout, diagnostic=lambda _, moments: {'probe_all_finite': True, 'channel_mse': .1})
    row = runner.evaluate(cfg, state, functions, tmp_path, target_step=1000)
    assert row['env_steps'] == cfg.steps_per_update and row['target_step'] == 1000
    assert row['stochastic/spatial/agent_center_distance_xy_mean'] == 2.
    assert row['potential/potential_endpoint_distance_xy_mean'] == 2.
    assert row['stochastic/spatial/potential_endpoint_early_termination_count'] == 1
    assert row['heldout/channel_mse'] == .1 and row['evaluation_transitions'] == 12920
    assert calls[0][0] == tmp_path/'evaluations'/'step_000000001000'
    assert calls[0][2]['checkpoint_step'] == cfg.steps_per_update
    assert calls[0][2]['reset_preset'] == 'past_center_ball' and not calls[0][2]['skill_phase']
    assert (tmp_path/'eval.csv').exists()
    saved = runner.evaluation_checkpoint(cfg, state)
    assert set(saved) == {'ppo_params', 'observation_norm', 'estimator_params', 'env_steps'}
    assert int(saved['env_steps']) == cfg.steps_per_update


def test_normalized_ant_declares_reference_reset():
    _, cfg = resolve('configs/ant/latentgce.yaml')
    assert cfg.reset_preset == 'past_center_ball'


def test_training_saves_recoverable_and_compact_checkpoints_before_failed_evaluation(tmp_path, monkeypatch):
    cfg = replace(recipe(max_updates=1), run_dir=str(tmp_path/'run'))
    state = dict(update=0, ppo=SimpleNamespace(params={'actor': jnp.asarray(2.)}),
                 controller=SimpleNamespace(observation_norm=empty_observation_norm(42)),
                 estimator=SimpleNamespace(params={'channel': jnp.asarray(3.)}),
                 buffer='training replay')
    events = []

    class Store:
        def __init__(self, root, **kwargs):
            self.root = root

        def save(self, step, payload, config, **kwargs):
            events.append(('save', self.root.name, step, set(payload), config))

    def update(original, payload, functions):
        payload['update'] += 1
        return {'env_steps': payload['update']*original.steps_per_update}

    def evaluate(original, payload, functions, output, *, target_step, mirror):
        events.append(('evaluate', target_step))
        if target_step:
            raise RuntimeError('deliberate media failure')
        return {'env_steps': 0}

    monkeypatch.setattr(runner.jax, 'devices', lambda: [SimpleNamespace(platform='gpu', device_kind='test stub')])
    monkeypatch.setattr(runner, 'build', lambda _: (state, {'environment_config': {'ctrl_dt': .1}}))
    monkeypatch.setattr(runner, 'CheckpointStore', Store)
    monkeypatch.setattr(runner, 'update', update)
    monkeypatch.setattr(runner, 'evaluate', evaluate)
    monkeypatch.setattr(runner, 'version', lambda _: 'test')
    monkeypatch.setattr(runner.WandbMirror, 'start', lambda *args, **kwargs: SimpleNamespace(
        log=lambda row: None, finish=lambda: None))
    with pytest.raises(RuntimeError, match='deliberate media failure'):
        runner.run(cfg)
    assert [(event[0], event[1], event[2] if len(event) > 2 else None) for event in events] == [
        ('save', 'checkpoints', 0), ('save', 'evaluation_checkpoints', 0), ('evaluate', 0, None),
        ('save', 'checkpoints', cfg.steps_per_update),
        ('save', 'evaluation_checkpoints', cfg.steps_per_update),
        ('evaluate', cfg.steps_per_update, None),
    ]
    assert 'buffer' in events[-3][3] and 'buffer' not in events[-2][3]
    assert events[-3][4]['evaluation_target_steps'] == [cfg.steps_per_update]


def test_full_run_executes_final_evaluation_at_reached_budget(tmp_path, monkeypatch):
    cfg = replace(recipe(), total_env_steps=279, run_dir=str(tmp_path/'run'))
    state = dict(update=0, ppo=SimpleNamespace(params={}),
                 controller=SimpleNamespace(observation_norm=empty_observation_norm(42)),
                 estimator=SimpleNamespace(params={}))
    events = []

    class Store:
        def __init__(self, *args, **kwargs):
            pass

        def save(self, *args, **kwargs):
            pass

    def update(original, payload, functions):
        payload['update'] += 1
        return {'env_steps': payload['update']*original.steps_per_update}

    def evaluate(original, payload, functions, output, *, target_step, mirror):
        step = payload['update']*original.steps_per_update
        events.append((target_step, step))
        return {'env_steps': step, 'target_step': target_step, 'measured_final_metric': float(step)}

    monkeypatch.setattr(runner.jax, 'devices', lambda: [SimpleNamespace(platform='gpu', device_kind='test stub')])
    monkeypatch.setattr(runner, 'build', lambda _: (state, {'environment_config': {}}))
    monkeypatch.setattr(runner, 'CheckpointStore', Store)
    monkeypatch.setattr(runner, 'update', update)
    monkeypatch.setattr(runner, 'evaluate', evaluate)
    monkeypatch.setattr(runner, 'version', lambda _: 'test')
    monkeypatch.setattr(runner.WandbMirror, 'start', lambda *args, **kwargs: SimpleNamespace(
        log=lambda row: None, finish=lambda: None))
    runner.run(cfg)
    reached = cfg.updates*cfg.steps_per_update
    assert reached == 272 and cfg.total_env_steps == 279
    assert events == [(0, 0), (reached, reached)]
    import json
    final = json.loads((tmp_path/'run'/'metrics.jsonl').read_text().splitlines()[-1])
    assert final['target_step'] == reached
    assert final['measured_final_metric'] == reached
