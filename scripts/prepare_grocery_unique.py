"""Derive a fixed 99-unique-negative Grocery set from ReChorus positives."""

import argparse
import ast
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def _read(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def prepare(source, output, negative_count=99, seed=2026):
    source, output = Path(source), Path(output)
    tables = {phase: _read(source / f'{phase}.csv')
              for phase in ('train', 'dev', 'test')}
    all_rows = [row for table in tables.values() for row in table]
    catalog = sorted({int(row['item_id']) for row in all_rows})
    clicked = defaultdict(set)
    for row in all_rows:
        clicked[int(row['user_id'])].add(int(row['item_id']))
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / 'train.csv', output / 'train.csv')
    duplicates_in_source = {}
    for offset, phase in enumerate(('dev', 'test')):
        duplicate_rows = 0
        with (output / f'{phase}.csv').open('w', encoding='utf-8', newline='') as out:
            writer = csv.writer(out, delimiter='\t')
            writer.writerow(['user_id', 'item_id', 'time', 'neg_items'])
            for row in tables[phase]:
                original = ast.literal_eval(row['neg_items'])
                duplicate_rows += len(original) != len(set(original))
                user = int(row['user_id'])
                pool = [item for item in catalog if item not in clicked[user]]
                if len(pool) < negative_count:
                    raise ValueError(f'User {user} lacks enough unseen items')
                negatives = random.Random(seed + 2 * user + offset).sample(
                    pool, negative_count)
                writer.writerow([user, int(row['item_id']), int(row['time']),
                                 str(negatives)])
        duplicates_in_source[phase] = duplicate_rows
    metadata = {'source': str(source), 'negative_count': negative_count,
                'seed': seed, 'duplicate_negative_rows_in_source': duplicates_in_source,
                'positive_rows_unchanged': True}
    (output / 'preparation.json').write_text(json.dumps(metadata, indent=2),
                                             encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--negatives', type=int, default=99)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    print(prepare(args.source, args.output, args.negatives, args.seed))
