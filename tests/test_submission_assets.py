import hashlib
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest

from scripts import build_submission_assets as assets


def test_existing_bank_is_verified_without_training_or_rebuilding(tmp_path, monkeypatch):
    target = tmp_path/'artifacts'/assets.BANK_NAME
    target.mkdir(parents=True)
    (target/'preserve.txt').write_text('frozen')
    validate = Mock()
    run = Mock()
    monkeypatch.setattr(assets, 'validate_bank', validate)
    monkeypatch.setattr(assets.subprocess, 'run', run)
    assert assets.ensure(tmp_path) == target
    validate.assert_called_once_with(target)
    run.assert_not_called()
    assert (target/'preserve.txt').read_text() == 'frozen'


def test_missing_or_unrecognized_training_data_cannot_start_a_build(tmp_path, monkeypatch):
    run = Mock()
    monkeypatch.setattr(assets.subprocess, 'run', run)
    with pytest.raises(ValueError, match='original training recording'):
        assets.ensure(tmp_path)
    (tmp_path/assets.SOURCE_NAME).write_bytes(b'test recording, not the training source')
    with pytest.raises(ValueError, match='SHA-256 differs'):
        assets.ensure(tmp_path)
    run.assert_not_called()
    assert not (tmp_path/'artifacts').exists()


def test_recipe_uses_declared_fitting_windows_and_existing_builders():
    import build_map
    from camloc import contracts as c

    planned = assets.commands(Path('training.mcap'), Path('work'))
    args = []
    # Parse through the actual builder CLI: protect against drift in option names/defaults.
    original = build_map.build
    try:
        build_map.build = args.append
        for command in planned[:2]:
            build_map.main(command[4:])
    finally:
        build_map.build = original
    fit = c.load_split()['fit']
    for config in args:
        assert not config.skip_bag_hash and not config.spatial_features
        for a, b in config.window:
            assert 0 <= a < b <= (fit.end_ns-fit.start_ns)/c.NS_PER_S
    assert args[0].window == [[0, 700.39]] and args[0].cameras == ['fr']
    assert args[1].window == [[265, 274], [299, 309], [372, 379]]
    assert args[1].rescue_masks and args[1].pair_target_baseline_m == 12
    assert set(args[1].cameras) == set(c.CAMERA_IDS)-{'fr'}
    assert '--verify-bag-hash' in planned[-1]
    assert all('reference_reconciled' not in str(command) and 'run.py' not in str(command)
               for command in planned)


@pytest.mark.parametrize('fail', [False, True])
def test_only_a_successfully_validated_build_is_published(tmp_path, monkeypatch, fail):
    data = b'original fixture'
    (tmp_path/assets.SOURCE_NAME).write_bytes(data)
    monkeypatch.setattr(assets, 'SOURCE_SHA256', hashlib.sha256(data).hexdigest())
    target = tmp_path/'artifacts'/assets.BANK_NAME
    work = tmp_path/'artifacts/submission-map-build'
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs == {'cwd': tmp_path, 'check': True}
        assert not target.exists()
        if len(calls) == 2 and fail:
            raise subprocess.CalledProcessError(1, command)
        if '--output' in command:
            Path(command[command.index('--output')+1]).mkdir()

    validate = Mock()
    monkeypatch.setattr(assets.subprocess, 'run', run)
    monkeypatch.setattr(assets, 'validate_bank', validate)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            assets.ensure(tmp_path)
        assert not target.exists()
        validate.assert_not_called()
    else:
        assert assets.ensure(tmp_path) == target
        assert len(calls) == 4 and target.is_dir()
        validate.assert_called_once_with(work/'combined')
    with pytest.raises(ValueError, match='previous build'):
        # A failed/staged build is never silently removed or overwritten.
        if target.exists():
            target.rename(tmp_path/'preserved-bank')
        assets.ensure(tmp_path)
