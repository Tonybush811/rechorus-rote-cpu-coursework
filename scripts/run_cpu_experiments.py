"""Run the fixed CPU comparison protocol and preserve every ReChorus log."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('Grocery_RoTE_99Unique', 'ML_1MTOPK_RoTE')
MODELS = ('RoTE', 'SASRec', 'GRU4Rec')
METRICS = ('HR@5', 'NDCG@5', 'HR@10', 'NDCG@10')
UPSTREAM_COMMIT = 'c164ec4303cc20ddcfbd1b57de366a481811d1e5'


def parse_log(contents):
    parameters = re.search(r'#params:\s*(\d+)', contents)
    best = re.search(r'Best Iter\(dev\)=\s*(\d+)', contents)
    result = re.search(r'Test After Training:\s*\(([^)]+)\)', contents)
    if not all((parameters, best, result)):
        raise ValueError('ReChorus log lacks parameters, best epoch, or final test result')
    pairs = dict((name, float(value)) for name, value in
                 re.findall(r'([A-Z]+@\d+):([0-9.]+)', result.group(1)))
    if set(pairs) != set(METRICS):
        raise ValueError(f'Expected {METRICS}, got {sorted(pairs)}')
    return {'parameters': int(parameters.group(1)),
            'best_epoch': int(best.group(1)), 'metrics': pairs}


def run_one(mode, dataset, model, seed):
    tag = f'{dataset}__{model}__seed{seed}'
    root = ROOT / 'results'
    record = root / 'runs' / mode / f'{tag}.json'
    if record.exists():
        data = json.loads(record.read_text(encoding='utf-8'))
        if data.get('status') == 'complete':
            print(f'SKIP {tag}', flush=True)
            return
    log_file = root / 'logs' / mode / f'{tag}.txt'
    output_file = root / 'stdout' / mode / f'{tag}.txt'
    checkpoint = root / 'checkpoints' / mode / f'{tag}.pt'
    for path in (record, log_file, output_file, checkpoint):
        path.parent.mkdir(parents=True, exist_ok=True)
    epoch = 1 if mode == 'smoke' else 8
    command = [sys.executable, 'src/main.py', '--model_name', model,
               '--dataset', dataset, '--gpu', '-1', '--num_workers', '0',
               '--random_seed', str(seed), '--history_max', '20',
               '--emb_size', '16', '--batch_size', '256',
               '--eval_batch_size', '256', '--lr', '0.001', '--l2', '0',
               '--epoch', str(epoch), '--early_stop', '3',
               '--topk', '5,10', '--metric', 'NDCG,HR',
               '--main_metric', 'NDCG@10', '--save_final_results', '0',
               '--log_file', log_file.as_posix(),
               '--model_path', checkpoint.as_posix()]
    if model == 'GRU4Rec':
        command += ['--hidden_size', '16']
    else:
        command += ['--num_layers', '1', '--num_heads', '2']
    print(f'RUN {tag} ({epoch} epochs max)', flush=True)
    started = time.perf_counter()
    with output_file.open('w', encoding='utf-8', errors='replace') as output:
        finished = subprocess.run(command, cwd=ROOT, stdout=output,
                                  stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - started
    if finished.returncode:
        tail = output_file.read_text(encoding='utf-8', errors='replace')[-2000:]
        raise RuntimeError(f'{tag} exited {finished.returncode}:\n{tail}')
    parsed = parse_log(log_file.read_text(encoding='utf-8', errors='replace'))
    record.write_text(json.dumps({
        'status': 'complete', 'framework_commit': UPSTREAM_COMMIT,
        'mode': mode, 'dataset': dataset, 'model': model, 'seed': seed,
        'epochs_max': epoch, 'elapsed_seconds': round(seconds, 2),
        'command': command, **parsed}, indent=2), encoding='utf-8')
    print(f'DONE {tag} NDCG@10={parsed["metrics"]["NDCG@10"]:.4f} '
          f'in {seconds:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'full'), required=True)
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    datasets = (args.dataset,) if args.dataset else DATASETS
    models = (args.model,) if args.model else MODELS
    seeds = (args.seed,) if args.seed is not None else ((0,) if args.mode == 'smoke' else (0, 1, 2))
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                run_one(args.mode, dataset, model, seed)
