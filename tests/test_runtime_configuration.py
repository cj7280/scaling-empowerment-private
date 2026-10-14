"""Machine settings and errors must not select hidden runtime alternatives."""
from pathlib import Path

import pytest

from empowerment_warp.tracking import _reserved_identity
from empowerment_warp._vendor.publication import PLOTTING_ROOT
from empowerment_warp._vendor.plotting.plot_style import materialize_pub_font


@pytest.mark.parametrize('requested,expected', [('online', 'online'), ('disabled', 'disabled'), (None, 'offline')])
def test_explicit_wandb_mode_overrides_local_default(tmp_path, requested, expected):
    import os
    import shutil
    import subprocess
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    shutil.copy2(Path(__file__).parents[1] / 'scripts/env.sh', scripts / 'env.sh')
    (tmp_path / '.env').write_text('PYTHON=/usr/bin/python3\nWANDB_MODE=offline\n')
    env = dict(os.environ)
    env.pop('WANDB_MODE', None)
    if requested is not None:
        env['WANDB_MODE'] = requested
    result = subprocess.check_output(
        ['bash', '-c', 'source "$1"; printf "%s" "$WANDB_MODE"', '_', str(scripts / 'env.sh')],
        env=env, text=True,
    )
    assert result == expected


def test_wandb_identity_comes_from_environment_or_saved_run(monkeypatch, tmp_path):
    monkeypatch.delenv('WANDB_ENTITY', raising=False)
    identity = _reserved_identity(tmp_path, {}, {}, 'disabled')
    assert identity['entity'] is None and identity['url'] is None
    monkeypatch.setenv('WANDB_ENTITY', 'configured-team')
    assert _reserved_identity(tmp_path, {}, {}, 'offline')['entity'] == 'configured-team'
    assert _reserved_identity(tmp_path, {}, {'entity': 'original-team'}, 'online')['entity'] == 'original-team'


@pytest.mark.parametrize('bold,variable', [(False, 'PLOTTING_REGULAR_FONT'), (True, 'PLOTTING_BOLD_FONT')])
def test_font_override_is_used_and_missing_font_raises(monkeypatch, tmp_path, bold, variable):
    monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv('PLOTTING_FONT_ROOT', raising=False)
    bundled = materialize_pub_font(bold=bold)
    assert Path(bundled).parent == PLOTTING_ROOT / 'fonts'
    configured = tmp_path / Path(bundled).name
    configured.write_bytes(Path(bundled).read_bytes())
    monkeypatch.setenv(variable, str(configured))
    assert materialize_pub_font(bold=bold) == str(configured)
    configured.unlink()
    with pytest.raises(FileNotFoundError, match=variable):
        materialize_pub_font(bold=bold, fallback_path=bundled)


def test_package_import_does_not_publish_plotting_modules():
    import subprocess
    import sys
    code = '''
import sys
before = list(sys.path)
import empowerment_warp
assert sys.path == before
assert not ({"plot_style", "plot_nodes", "contracts", "video_render"} & sys.modules.keys())
from empowerment_warp._vendor.plotting import plot_style
assert sys.path == before
assert "plot_style" not in sys.modules
'''
    subprocess.run([sys.executable, "-c", code], check=True)
