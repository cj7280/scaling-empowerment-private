"""Each named ablation must change the intended scientific choice."""
from pathlib import Path
import pytest
from empowerment_warp.recipes import resolve

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('env', ['ant','ant_soccer','builderbench_corner3','craftax_classic'])
def test_structural_objectives_seeds_and_data_sources(env):
    names = ['see', 'joint_mi' if env == 'craftax_classic' else 'cmi', 'no_switch', 'pair_decoder']
    resolved = {}
    for name in names:
        recipe, config = resolve(ROOT / 'configs' / env / (name + '.yaml'))
        assert recipe['seeds'] == list(range(5 if name == 'see' else 3))
        assert config.encoder_objective == 'forward_infonce'
        assert config.continuous_reward_normalization == 'sampled_infonce'
        resolved[name] = recipe['settings']
    if env == 'craftax_classic':
        assert resolved['see']['encoder_data_source'] == 'branches'
        assert resolved['joint_mi']['encoder_data_source'] == 'live'
        for name in ['pair_decoder','no_switch']:
            assert resolved[name]['encoder_data_source'] == 'branches'
        assert resolved['joint_mi']['potential_steps'] == resolved['see']['potential_steps'] > 0
        changed = {k for k in resolved['see'] if resolved['see'][k] != resolved['joint_mi'][k]}
        assert changed == {'encoder_data_source'}
        changed = {k for k in resolved['see'] if resolved['see'][k] != resolved['pair_decoder'][k]}
        assert changed == {'pair_form'}
        changed = {k for k in resolved['see'] if resolved['see'][k] != resolved['no_switch'][k]}
        assert changed == {'potential_steps', 'potential_lane_fraction', 'encoder_branch_steps'}
        _, base = resolve(ROOT / 'configs' / env / 'see.yaml')
        _, no_switch = resolve(ROOT / 'configs' / env / 'no_switch.yaml')
        assert base.branch_horizon_steps == no_switch.branch_horizon_steps == 900
        assert base.effective_recipe()['branch_physical_steps_per_live_transition'] == no_switch.effective_recipe()['branch_physical_steps_per_live_transition'] == .9
    else:
        assert resolved['cmi']['branch_live_two_phase']
        assert resolved['cmi']['skill_start_mode'] == 'marginal_branch'
        assert resolved['no_switch']['potential_steps'] == 0
        assert resolved['no_switch']['potential_lane_fraction'] == .25
        assert resolved['pair_decoder']['pair_form'] == 'pair_mlp'
