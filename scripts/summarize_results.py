"""Validate and aggregate the fixed 2 x 3 x 3 experiment grid."""

import csv
import json
import statistics
from pathlib import Path

from run_cpu_experiments import DATASETS, METRICS, MODELS, ROOT


def summarize(records_dir):
    records_dir = Path(records_dir)
    summary = []
    for dataset in DATASETS:
        for model in MODELS:
            records = []
            for seed in (0, 1, 2):
                path = records_dir / f'{dataset}__{model}__seed{seed}.json'
                record = json.loads(path.read_text(encoding='utf-8'))
                if (record['status'] != 'complete' or record['mode'] != 'full' or
                        record['dataset'] != dataset or record['model'] != model or
                        record['seed'] != seed or set(record['metrics']) != set(METRICS)):
                    raise ValueError(f'Invalid record: {path}')
                records.append(record)
            row = {'dataset': dataset, 'model': model, 'seeds': [0, 1, 2]}
            for metric in METRICS:
                values = [record['metrics'][metric] for record in records]
                row[metric] = {'mean': statistics.mean(values),
                               'std': statistics.stdev(values)}
            for label, values in (
                    ('elapsed_seconds', [r['elapsed_seconds'] for r in records]),
                    ('best_epoch', [r['best_epoch'] for r in records]),
                    ('parameters', [r['parameters'] for r in records])):
                row[label] = {'mean': statistics.mean(values),
                              'std': statistics.stdev(values)}
            summary.append(row)
    return summary


def write_summary(rows, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'summary.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    columns = ['dataset', 'model', 'seeds'] + [
        f'{name}_{measure}' for name in (*METRICS, 'elapsed_seconds',
                                        'best_epoch', 'parameters')
        for measure in ('mean', 'std')]
    with (destination / 'summary.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {'dataset': row['dataset'], 'model': row['model'],
                    'seeds': ','.join(map(str, row['seeds']))}
            for name in (*METRICS, 'elapsed_seconds', 'best_epoch', 'parameters'):
                for measure in ('mean', 'std'):
                    flat[f'{name}_{measure}'] = row[name][measure]
            writer.writerow(flat)


if __name__ == '__main__':
    results = summarize(ROOT / 'results' / 'runs' / 'full')
    write_summary(results, ROOT / 'results')
    for row in results:
        print(row['dataset'], row['model'],
              'NDCG@10', f"{row['NDCG@10']['mean']:.4f} ± {row['NDCG@10']['std']:.4f}")
