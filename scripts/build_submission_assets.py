"""Build the submission map from the original recording's fixed fitting windows."""
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camloc import contracts as c, landmark_map as lm, retrieval as rt  # noqa: E402

SOURCE_SHA256 = 'fe5a0b9565b1bc03dd1ad775645722645e135b2f19012cf7bc815a8bcb91d613'
SOURCE_NAME = 'c0_2026-10-13_22-53-42_camloc_0.mcap'
BANK_NAME = 'fit-seven-rescue-v2'


def commands(bag, work):
    front, rescue, combined = (work/name for name in ('front', 'rescue', 'combined'))
    base = [sys.executable, '-u', '-B', str(ROOT/'build_map.py'), '--bag', str(bag)]
    return [
        [*base, '--output', str(front), '--cameras', 'fr', '--window', '0', '700.39',
         '--travel-m', '4', '--heading-deg', '5', '--neighbors', '8'],
        [*base, '--output', str(rescue), '--cameras', 'fl', 'cl', 'cr', 'r', 'rl', 'rr',
         '--window', '265', '274', '--window', '299', '309', '--window', '372', '379',
         '--travel-m', '5', '--heading-deg', '10', '--neighbors', '8',
         '--rescue-masks', '--pair-target-baseline-m', '12'],
        [sys.executable, '-u', '-B', str(ROOT/'scripts/merge_landmark_banks.py'),
         '--base', str(front), '--rescue', str(rescue), '--output', str(combined)],
        [sys.executable, '-u', '-B', str(ROOT/'scripts/build_retrieval_index.py'),
         '--artifacts', str(combined), '--bag', str(bag), '--verify-bag-hash'],
    ]


def validate_bank(path):
    bank = lm.load_landmark_map(path)
    rt.RetrievalIndex.load(path, bank, None)
    source = bank.manifest['source_bag']
    if (source['name'] != SOURCE_NAME or source['sha256'] != SOURCE_SHA256
            or bank.manifest['split_sha256'] != c.BUILD_SPLIT_SHA256
            or bank.manifest.get('camera_policy') != 'support-v1'
            or set(bank.cameras) != set(c.CAMERA_IDS)):
        raise ValueError('Submission bank differs from the declared seven-camera fitting recipe')
    print(f'Verified submission map: {bank.map_id}', flush=True)


def ensure(root=ROOT):
    root = Path(root).resolve()
    artifacts = (root/'artifacts').resolve()
    target = (artifacts/BANK_NAME).resolve()
    work = (artifacts/'submission-map-build').resolve()
    if not artifacts.is_relative_to(root) or not all(p.is_relative_to(artifacts) for p in (target, work)):
        raise ValueError('Build outputs must stay inside the project artifacts directory')
    c.load_split()  # Verify the immutable manifest even when reusing a supplied bank.
    if target.exists():
        validate_bank(target)
        return target
    bag = root/SOURCE_NAME
    if not bag.is_file():
        raise ValueError(f'Place the original training recording at {bag}. Test/holdout clips cannot build the map.')
    with bag.open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != SOURCE_SHA256:
            raise ValueError('Training recording SHA-256 differs; refusing to fit on another dataset')
    if work.exists():
        raise ValueError(f'A previous build exists at {work}; preserve it and rename it before retrying.')
    work.mkdir(parents=True)
    print('Building fitting-only submission assets. This can take many minutes on a GPU.', flush=True)
    for number, command in enumerate(commands(bag, work), 1):
        print(f'Map build stage {number}/4: {Path(command[3]).name}', flush=True)
        subprocess.run(command, cwd=root, check=True)
    combined = (work/'combined').resolve()
    if not combined.is_relative_to(artifacts):
        raise ValueError('Combined bank is outside project artifacts')
    validate_bank(combined)
    combined.rename(target)
    print(f'Submission map ready: {target}. Rebuilt banks require their own accuracy evaluation.', flush=True)
    return target


if __name__ == '__main__':
    try:
        ensure()
    except (ValueError, OSError, c.ContractError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'Submission map preparation failed: {error}') from error
