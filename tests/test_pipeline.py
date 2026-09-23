"""End-to-end CPU integration through ReChorus Reader, Model and Runner."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('model', ['RoTE', 'SASRec', 'GRU4Rec'])
def test_one_epoch_cpu_pipeline(model, tmp_path: Path):
    directory = tmp_path / 'tiny'
    directory.mkdir()
    splits = {'train': [], 'dev': [], 'test': []}
    for user in range(1, 21):
        items = [((user * 7 + step) % 110) + 1 for step in range(8)]
        available = [item for item in range(1, 111) if item not in items]
        for step, item in enumerate(items):
            phase = 'train' if step < 6 else ('dev' if step == 6 else 'test')
            row = [user, item, 1_000_000 + user * 10 + step]
            if phase != 'train':
                row.append(str(available[:99]))
            splits[phase].append(row)
    for phase, rows in splits.items():
        with (directory / f'{phase}.csv').open('w', encoding='utf-8', newline='') as out:
            writer = csv.writer(out, delimiter='\t')
            writer.writerow(['user_id', 'item_id', 'time'] +
                            ([] if phase == 'train' else ['neg_items']))
            writer.writerows(rows)
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, 'src/main.py', '--model_name', model,
               '--path', str(tmp_path), '--dataset', 'tiny', '--gpu', '-1',
               '--num_workers', '0', '--epoch', '1', '--history_max', '4',
               '--emb_size', '8', '--batch_size', '16', '--eval_batch_size', '16',
               '--save_final_results', '0', '--log_file', str(tmp_path / 'run.log'),
               '--model_path', str(tmp_path / 'model.pt')]
    if model == 'GRU4Rec':
        command += ['--hidden_size', '8']
    else:
        command += ['--num_heads', '2', '--num_layers', '1']
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    assert 'Test After Training:' in completed.stdout
