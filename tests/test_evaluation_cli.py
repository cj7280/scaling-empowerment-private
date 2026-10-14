"""Evaluation dispatch and source-run protection through the public CLI."""
import json
from empowerment_warp.cli import main
import pytest


def test_pqn_saved_algorithm_dispatches_and_preserves_source(tmp_path, monkeypatch):
    from empowerment_warp.training.pqn import branch_eval
    source = tmp_path / 'source'
    source.mkdir()
    config = {'rl_algorithm': 'potential-pqn', 'environment': 'craftax:classic-symbolic'}
    (source / 'config.json').write_text(json.dumps(config))
    checkpoint = source / 'checkpoints/params_000000000123.pkl'
    checkpoint.parent.mkdir()
    checkpoint.touch()
    output = tmp_path / 'evaluation'
    calls = []
    def evaluate(run_dir, path, **kwargs):
        calls.append((run_dir,path,kwargs))
        return {'env_steps':123}
    monkeypatch.setattr(branch_eval, 'evaluate_checkpoint', evaluate)
    main(['evaluate', str(source), '--output', str(output), '--contexts', '2'])
    assert calls[0][0:2] == (source,checkpoint)
    assert calls[0][2]['contexts'] == 2
    assert json.loads((output/'metrics.json').read_text()) == {'env_steps':123}
    assert json.loads((source/'config.json').read_text()) == config


def test_evaluation_cannot_write_inside_source_run(tmp_path):
    with pytest.raises(ValueError, match='outside the source'):
        main(['evaluate',str(tmp_path),'--output',str(tmp_path/'evaluation')])


def test_sac_checkpoint_and_target_are_forwarded(tmp_path, monkeypatch):
    from empowerment_warp import evaluation_protocol
    source = tmp_path / 'source'; source.mkdir()
    (source/'config.json').write_text(json.dumps({'algorithm':'potential','environment':'ant'}))
    calls=[]
    monkeypatch.setattr(evaluation_protocol, 'evaluate_sac',
        lambda *args, **kwargs: calls.append(kwargs) or {})
    main(['evaluate',str(source),'--output',str(tmp_path/'out'),
          '--checkpoint','saved/step_000000000000','--target-step','0'])
    assert calls == [{'checkpoint':'saved/step_000000000000','target_step':0}]


@pytest.mark.parametrize('environment', ['ant', 'builderbench_corner3'])
def test_branch_media_retained_checkpoint_is_forwarded(tmp_path, monkeypatch, environment):
    from empowerment_warp.media import locomotion, builderbench
    source = tmp_path / 'source'; source.mkdir()
    (source/'config.json').write_text(json.dumps({'environment': environment}))
    calls = []
    module, function = ((builderbench, 'create_builderbench_branch_media')
                        if environment == 'builderbench_corner3'
                        else (locomotion, 'create_locomotion_branch_media'))
    monkeypatch.setattr(module, function, lambda *a, **kw: calls.append(kw) or {})
    main(['branch-media', str(source), '--output', str(tmp_path/'media'),
          '--checkpoint', 'saved/step_000000000000'])
    assert calls[0]['checkpoint'] == 'saved/step_000000000000'


def test_recipe_is_saved_before_training_starts(tmp_path, monkeypatch):
    from pathlib import Path
    from empowerment_warp.training.sac import runner
    root = Path(__file__).parents[1]
    output = tmp_path / 'new-run'
    def fail_after_check(config, **kwargs):
        saved = json.loads((output / 'recipe.json').read_text())
        assert saved['settings']['run_dir'] == str(output)
        raise RuntimeError('controlled startup failure')
    monkeypatch.setattr(runner, 'train', fail_after_check)
    with pytest.raises(RuntimeError, match='controlled startup failure'):
        main(['train', '--config', str(root / 'configs/ant/see.yaml'), '--run-dir', str(output)])
    assert (output / 'recipe.json').is_file()
