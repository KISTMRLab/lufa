"""Start the LUFA viewer: learned retrieval when a local checkpoint and bank exist, else the authored example.

Default (ignored) locations:
  outputs/lufa/model/      checkpoint directory from `lufa train` (model.pt + tokenizer/)
  outputs/lufa/bank.npz    recorded-motion bank from `lufa bank`
`--train-small` prepares BEAT takes found under --beat-root (default outputs/beat-raw), trains a compact
from-scratch model for a few epochs on CPU, builds the bank and then launches learned retrieval offline.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--model', default='outputs/lufa/model', help='checkpoint directory (default: %(default)s)')
    parser.add_argument('--bank', default='outputs/lufa/bank.npz', help='bank file (default: %(default)s)')
    parser.add_argument('--authored', action='store_true', help='always start with the authored controller example')
    parser.add_argument('--train-small', action='store_true',
                        help='prepare local BEAT takes, train a small from-scratch model and build the bank first')
    parser.add_argument('--beat-root', default='outputs/beat-raw',
                        help='BEAT takes (.wav/.json/.TextGrid) for --train-small (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--device', default='cpu')
    return parser.parse_args(argv)


def serve_command(model, bank, port, authored=False, device='cpu'):
    """(command, learned?) — learned retrieval only when both the checkpoint and the bank exist."""
    model, bank = Path(model), Path(bank)
    learned = not authored and (model / 'model.pt').is_file() and bank.is_file()
    command = [sys.executable, '-m', 'lufa.cli', 'serve', '--port', str(port)]
    if learned:
        command += ['--model', str(model), '--bank', str(bank), '--device', device]
    return command, learned


def lufa(*command):
    subprocess.run([sys.executable, '-m', 'lufa.cli', *command], check=True)


def train_small(beat_root, model, bank, epochs, device):
    beat_root, model, bank = Path(beat_root), Path(model), Path(bank)
    if not beat_root.is_dir() or not any(beat_root.rglob('*.TextGrid')):
        raise SystemExit(f'No BEAT takes under {beat_root}. Fetch a small official sample first, e.g.\n'
                         f'  lufa fetch-beat --speaker 1 --takes 1_wayne_0_1_1 1_wayne_0_2_2 1_wayne_0_3_3 --output {beat_root}')
    data = model.parent / 'small-data'
    # With one local speaker every window is a training record; more speakers get a speaker-disjoint split.
    lufa('prepare-beat', '--root', str(beat_root), '--output', str(data), '--hop', '1.5')
    lufa('train', '--manifest', str(data / 'manifest.jsonl'), '--from-scratch', '--corpus-vocab', '--epochs', str(epochs),
         '--batch-size', '8', '--lr', '3e-4', '--output', str(model), '--device', device)
    lufa('bank', '--manifest', str(data / 'manifest.jsonl'), '--model', str(model), '--output', str(bank), '--device', device)
    rows = [json.loads(line) for line in (data / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    train_rows = [r for r in rows if r['split'] == 'train']
    speakers = ', '.join(sorted({r['speaker'] for r in train_rows}))
    (model / 'demo-provenance.json').write_text(json.dumps({
        'summary': f'small from-scratch demo model ({epochs} epochs, {len(train_rows)} BEAT clips, speakers {speakers})',
        'source': str(beat_root), 'epochs': epochs, 'clips': len(train_rows),
        'note': 'Trained locally for the demo; not the paper model and not a benchmark result.'}, indent=2), encoding='utf-8')


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    os.environ['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')
    prepare = ROOT / 'scripts' / 'prepare_viewer.py'
    if prepare.exists() and not all((ROOT / 'static' / 'vendor' / name).is_file() for name in ('three.module.js', 'GLTFLoader.js', 'BufferGeometryUtils.js')):
        subprocess.run([sys.executable, str(prepare)], check=True)
    if args.train_small:
        train_small(args.beat_root, args.model, args.bank, args.epochs, args.device)
    command, learned = serve_command(args.model, args.bank, args.port, args.authored, args.device)
    if learned:
        print(f'Open http://127.0.0.1:{args.port}/ — learned retrieval from {args.model} over {args.bank}.', flush=True)
    else:
        print(f'Open http://127.0.0.1:{args.port}/ — authored controller example (no trained model at {args.model} '
              f'and bank at {args.bank}; run with --train-small or see the README BEAT-first setup).', flush=True)
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(main())
