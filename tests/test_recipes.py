"""Recipe inheritance, overrides, and shipped recipe settings."""
from pathlib import Path

import pytest
import yaml

from empowerment_warp.recipes import apply_overrides, read_recipe, resolve

ROOT = Path(__file__).resolve().parents[1]
RECIPES = sorted((ROOT / 'configs').glob('*/*.yaml'))
FIVE_SEED_RECIPES = {f'{env}/latentgce.yaml' for env in ('ant', 'ant_soccer', 'builderbench_corner3')}


def test_strict_overrides():
    settings = {'seed': 0, 'estimator': {'learning_rate': 0.001}}
    assert apply_overrides(settings, ['estimator.learning_rate=0.002'])['estimator']['learning_rate'] == 0.002
    assert settings['estimator']['learning_rate'] == 0.001
    with pytest.raises(ValueError, match='unknown override'):
        apply_overrides(settings, ['learning_rate=0.1'])


def test_no_inheritance_chain(tmp_path):
    base = {'engine': 'sac', 'seeds': [0], 'settings': {'environment': 'ant'}}
    (tmp_path / 'base.yaml').write_text(yaml.safe_dump(base))
    (tmp_path / 'child.yaml').write_text('extends: base.yaml\nsettings: {}\n')
    (tmp_path / 'grandchild.yaml').write_text('extends: child.yaml\nsettings: {}\n')
    with pytest.raises(ValueError, match='without further inheritance'):
        read_recipe(tmp_path / 'grandchild.yaml')


@pytest.mark.parametrize('path', RECIPES, ids=lambda p: str(p.relative_to(ROOT / 'configs')))
def test_all_recipes_resolve(path):
    recipe, config = resolve(path)
    five = path.stem in {'see', 'see_reset_memory'} or str(path.relative_to(ROOT / 'configs')) in FIVE_SEED_RECIPES
    assert recipe['seeds'] == list(range(5 if five else 3))
    assert not Path(recipe['settings'].get('run_dir', 'does-not-exist')).exists()


def test_craftax_reward_comparisons_use_cmi():
    base, _ = resolve(ROOT / 'configs/craftax_classic/see.yaml')
    assert base['settings']['encoder_data_source'] == 'branches'
    for name, normalization in [('lambda0', 'none'), ('lambda1', 'sampled_infonce')]:
        treatment, _ = resolve(ROOT / f'configs/craftax_classic/{name}.yaml')
        changed = {k for k in base['settings'] if base['settings'][k] != treatment['settings'][k]}
        assert changed <= {'continuous_reward_normalization', 'reward_partition_anneal_start_steps', 'reward_partition_anneal_end_steps'}
        assert treatment['settings']['continuous_reward_normalization'] == normalization
        assert treatment['settings']['reward_partition_anneal_end_steps'] == 0


@pytest.mark.parametrize('path', sorted((ROOT / 'configs/craftax_classic').glob('*.yaml')),
                         ids=lambda p: p.stem)
def test_craftax_uses_current_encoder_without_averaging(path):
    _, config = resolve(path)
    skill = config.skill_config(1345)
    assert skill.reward_encoder == 'live'
    assert skill.relabel_encoder == 'live'
    assert skill.target_tau == 1.0


def test_no_switch_keeps_latest_probability():
    recipe, _ = resolve(ROOT / 'configs/craftax_classic/no_switch.yaml')
    assert recipe['settings']['potential_lane_fraction'] == 0.25


@pytest.mark.parametrize('environment', ['ant', 'ant_soccer', 'builderbench_corner3', 'craftax_classic'])
def test_sampled_infonce_default_and_vmf_override(environment):
    path = ROOT / f'configs/{environment}/see.yaml'
    base, _ = resolve(path)
    assert base['settings']['encoder_objective'] == 'forward_infonce'
    assert base['settings']['continuous_reward_normalization'] == 'sampled_infonce'
    overrides = ['encoder_objective=vmf_forward', 'continuous_reward_normalization=sphere']
    recipe, config = resolve(path, overrides=overrides)
    assert config.encoder_objective == 'vmf_forward'
    changed = {k for k in base['settings'] if base['settings'][k] != recipe['settings'][k]}
    assert changed == {'encoder_objective', 'continuous_reward_normalization'}
    if environment == 'craftax_classic':
        assert config.skill_config(1345).encoder_objective == 'vmf_forward'


@pytest.mark.parametrize('env,method,preset', [
    ('ant', 'vim', 'past_center_ball'), ('ant', 'latentgce', 'past_center_ball'),
    ('ant_soccer', 'vim', 'closer_ball'), ('ant_soccer', 'latentgce', 'closer_ball'),
    ('builderbench_corner3', 'vim', 'standard'),
])
def test_baseline_reset_reaches_environment_constructor(env, method, preset, monkeypatch):
    from empowerment_warp.training.baselines import runner
    recipe, config = resolve(ROOT / f'configs/{env}/{method}.yaml')
    assert config.resolved()['reset_preset'] == preset
    monkeypatch.setattr(runner, 'make_environment', lambda config: config)
    actual = runner.make_benchmark_environment(config)
    assert actual.antsoccer_preset == preset


def test_retired_relabel_controls_load_only_when_inactive():
    from empowerment_warp.training.sac.config import ExperimentConfig
    _, config = resolve(ROOT / 'configs/ant/see.yaml')
    saved = config.to_dict()
    saved.update(achieved_skill_relabel_old_replay_only=False, achieved_skill_relabel_recent_window=0)
    assert ExperimentConfig.from_dict(saved).to_dict() == config.to_dict()
    with pytest.raises(ValueError, match='retired feature'):
        ExperimentConfig.from_dict({**saved, 'achieved_skill_relabel_old_replay_only': True})
