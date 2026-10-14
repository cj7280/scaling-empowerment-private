"""First-episode phase counts from compact, freshly sampled switching evaluations."""
from collections import namedtuple
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from empowerment_warp.training.pqn import branch_eval
from empowerment_warp.agents.skill_classifier import SkillConfig, sample_skills

GameState = namedtuple('GameState', 'player_position achievements')
State = namedtuple('State', 'game_state obs reward terminated truncated step_count rng die_at')

class TinyGrid:
    def __init__(self, count):
        self.num_envs = count
    def reset(self, keys):
        n = self.num_envs
        pos = jnp.zeros((n, 2), jnp.int32)
        return State(GameState(pos, jnp.zeros((n, 4), bool)), pos.astype(jnp.float32),
            jnp.zeros(n), jnp.zeros(n, bool), jnp.zeros(n, bool), jnp.zeros(n, jnp.int32),
            keys, jnp.where(jnp.arange(n) == 0, 1, 4))
    def step(self, state, actions):
        steps = state.step_count + 1
        pos = state.game_state.player_position.at[:, 0].add(actions)
        return state._replace(game_state=GameState(pos, jnp.arange(4)[None] < steps[:, None]),
            obs=pos.astype(jnp.float32), reward=jnp.ones(self.num_envs),
            terminated=steps >= state.die_at, step_count=steps)

class Agent:
    config = {'skill_dim': 2}
    def initial_recurrent_state(self, count):
        return jnp.zeros((count, 1))
    def sample_actions_recurrent_from_params(self, variables, obs, skills, anchors,
                                             recurrent, reset, key, epsilon, deterministic=False):
        active = jnp.linalg.norm(skills, axis=-1) > 0
        # First skill action must reset recurrence, and its anchor must be the switch state.
        action = jnp.where(active, 2, 1)
        action += jnp.where(active & (obs[:, 0] == 2) & ((reset == 0) | (anchors[:, 0] != 2)), 100, 0)
        return jnp.zeros((len(skills), 2)), action, recurrent + 1

def tiny_rollout():
    fn = branch_eval.make_switching_rollout(TinyGrid(2), TinyGrid(4), Agent(),
        prefix_steps=2, episode_length=4, skills_per_context=2)
    return {k: np.asarray(v) for k, v in fn({}, jnp.asarray(.1),
        jax.random.PRNGKey(1), jax.random.PRNGKey(2), jnp.tile(jnp.eye(2), (2,1))).items()}

def test_switching_includes_terminal_unlock_and_never_restarts_dead_prefix():
    a = tiny_rollout()
    np.testing.assert_array_equal(a['cells'][:, 0, 1], [0,1,1,1,1])
    np.testing.assert_array_equal(a['cells'][:, 2, 1], [0,1,2,4,6])
    np.testing.assert_array_equal(a['valid'][:, 0], [1,1,0,0,0])
    np.testing.assert_array_equal(a['valid'][:, 2], [1,1,1,1,1])
    np.testing.assert_array_equal(a['episode_lengths'], [1,1,4,4])
    np.testing.assert_array_equal(a['episode_returns'], [1,1,4,4])
    counts = branch_eval.phase_achievement_counts(a['initial_achievements'],
        a['prefix_achievements'], a['final_achievements'])
    np.testing.assert_array_equal(counts['potential'], [1,1,2,2])
    np.testing.assert_array_equal(counts['skill'], [0,0,2,2])
    np.testing.assert_array_equal(counts['total'], counts['potential'] + counts['skill'])

def test_counts_exclude_reset_unlocks_and_reject_lost_achievements():
    result = branch_eval.phase_achievement_counts([[1,0,0]], [[1,1,0]], [[1,1,1]])
    assert result['total'][0] == 2 and result['potential'][0] == result['skill'][0] == 1
    with pytest.raises(ValueError, match='persist'):
        branch_eval.phase_achievement_counts([[0,0]], [[1,0]], [[0,1]])

def test_checkpoint_sampling_fresh_matched_reproducible_and_not_video_library():
    env = 'craftax:classic-symbolic'
    config = SkillConfig('potential', observation_dim=2, skill_dim=16, discrete_skills=False)
    a = branch_eval.evaluation_keys(env, 0, 10_000_000)
    b = branch_eval.evaluation_keys(env, 0, 20_000_000)
    c = branch_eval.evaluation_keys(env, 1, 10_000_000)
    np.testing.assert_array_equal(a, branch_eval.evaluation_keys(env, 0, 10_000_000))
    for first, later, seed in zip(a,b,c):
        assert not np.array_equal(first,later) and not np.array_equal(first,seed)
    skills = np.asarray(sample_skills(config, a[2], 128))
    assert np.unique(skills, axis=0).shape[0] == 128
    np.testing.assert_allclose(np.linalg.norm(skills,axis=1), 1, atol=2e-7)
    assert not np.array_equal(skills, sample_skills(config,b[2],128))
    # Continuous environment indices are stable, so this RNG chain is fixed.
    root = jax.random.PRNGKey(20261004)
    for value in (0,0,10_000_000,0):
        root = jax.random.fold_in(root,value)
    np.testing.assert_array_equal(branch_eval.evaluation_keys('ant',0,10_000_000),jax.random.split(root,3))

def test_saves_phase_metrics_from_same_trajectories_without_an_extra_probe(tmp_path):
    arrays = tiny_rollout()
    columns = tuple((i, f'Achievements/test{i}') for i in range(4))
    class Driver:
        def evaluate(self, *args, **kwargs):
            return arrays, columns, branch_eval.evaluation_keys('craftax:classic-symbolic',2,17)
    config = SimpleNamespace(seed=2,environment='craftax:classic-symbolic',eval_contexts=2,eval_skills_per_context=2,
        eval_episode_length=0,episode_length=4,run_dir=str(tmp_path))
    agent = SimpleNamespace(config={'skill_dim':2}, env_steps=17, n_updates=1, exploration_epsilon=lambda:.1)
    row = branch_eval.evaluate_agent(config,agent,np.eye(2),env_steps=17,output_dir=tmp_path,driver=Driver())
    assert row['evaluation/achievement_count'] == 2.5
    assert row['evaluation/potential_policy/achievement_count'] == 1.5
    assert row['evaluation/skill_policy/new_achievement_count'] == 1
    assert row['evaluation/completed_episodes'] == 4
    assert row['protocol']['num_envs'] == 4
    trace = np.load(tmp_path/'figures/step_17/evaluation_trajectories.npz')
    np.testing.assert_array_equal(trace['total_achievement_counts'],[1,1,4,4])
    assert not (tmp_path/'figures/step_17/potential_policy_player_trace.npz').exists()

def test_floor_changes_are_separate_from_xy_displacement():
    cells = np.asarray([[[0,0,0],[0,0,0]],[[0,3,4],[1,5,5]],[[0,3,4],[1,5,5]]])
    summary = branch_eval.summarize_potential_policy_trace(cells,np.array([[1,1],[1,1],[0,0]],bool))
    assert summary['displacement_mean'] == 5
    assert summary['floor_changed_fraction'] == .5

def test_no_switch_media_uses_eval_prefix_without_changing_training_config(monkeypatch, tmp_path):
    from empowerment_warp.training.pqn.config import PotentialPQNRunConfig
    config = PotentialPQNRunConfig(environment='craftax:classic-symbolic', run_dir=str(tmp_path),
        potential_steps=0, potential_lane_fraction=.25, eval_contexts=2, eval_skills_per_context=2)
    arrays = tiny_rollout()
    class Driver:
        media_env = None
        def evaluate(self, *args, **kwargs):
            return arrays, tuple((i,f'Achievements/test{i}') for i in range(4)), \
                branch_eval.evaluation_keys(config.environment,0,17)
    def media_env(cfg):
        assert cfg is config and cfg.potential_steps == 0 and cfg.potential_lane_fraction == .25
        return object()
    def collect(env, actor, cfg, library, **kwargs):
        assert cfg['potential_steps'] == kwargs['warmup_steps'] == 100
        return []
    def figures(output, step, rollouts, cfg, *, potential_steps):
        assert cfg is config and potential_steps == 100
        return {'fps':24}
    monkeypatch.setattr(branch_eval,'NativeScalarEvalEnv',media_env)
    monkeypatch.setattr(branch_eval.reference,'_StatefulRecurrentEvalActor',lambda *a:None)
    monkeypatch.setattr(branch_eval.reference,'_collect_jax_game_branch_skill_rollouts',collect)
    monkeypatch.setattr(branch_eval,'write_branch_figures',figures)
    agent = SimpleNamespace(config={'skill_dim':2}, env_steps=17, n_updates=1, exploration_epsilon=lambda:.1)
    row = branch_eval.evaluate_agent(config,agent,np.eye(2),env_steps=17,output_dir=tmp_path,
                                    driver=Driver(),media=True)
    assert row['figures']['protocol']['prefix_steps'] == 100
    assert row['figures']['protocol']['contributes_to_metrics'] is False
    assert config.potential_steps == 0 and config.potential_lane_fraction == .25

def test_all_craftax_recipes_use_128_worlds_one_skill_and_existing_cadence():
    from pathlib import Path
    from empowerment_warp.recipes import resolve
    from empowerment_warp.training.pqn.config import PotentialPQNRunConfig
    root = Path(__file__).resolve().parents[1]
    defaults = PotentialPQNRunConfig(environment='craftax:classic-symbolic',run_dir='unused')
    assert (defaults.eval_contexts,defaults.eval_skills_per_context) == (128,1)
    for path in sorted((root/'configs/craftax_classic').glob('*.yaml')):
        _, cfg = resolve(path)
        assert (cfg.eval_contexts,cfg.eval_skills_per_context) == (128,1), path
        assert cfg.episode_length == 1000
        assert cfg.eval_every_segments * cfg.rollout_steps * cfg.num_envs == 10_092_544
        assert cfg.eval_media_interval == 50_000_000
        assert cfg.eval_media_skill_count == 4 and cfg.eval_fixed_reset_seed == 0
        assert cfg.keep_checkpoints == 0
        assert cfg.checkpoint_every_segments == cfg.eval_every_segments
