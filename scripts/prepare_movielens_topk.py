"""Convert MovieLens-1M into ReChorus Top-K leave-one-out splits."""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def prepare(raw_file, output_dir, min_count=5, negative_count=99, seed=2026):
    raw_file, output_dir = Path(raw_file), Path(output_dir)
    if min_count < 3 or negative_count < 1:
        raise ValueError('min_count must be >=3 and negative_count must be >=1')
    ratings = []
    with raw_file.open(encoding='latin-1') as handle:
        for row_number, line in enumerate(handle):
            user, item, rating, timestamp = line.strip().split('::')
            if int(rating) >= 4:
                ratings.append((int(user), int(item), int(timestamp), row_number))

    while True:
        users = Counter(row[0] for row in ratings)
        items = Counter(row[1] for row in ratings)
        reduced = [row for row in ratings if users[row[0]] >= min_count
                   and items[row[1]] >= min_count]
        if len(reduced) == len(ratings):
            break
        ratings = reduced

    sequences = defaultdict(list)
    for user, item, timestamp, row_number in ratings:
        sequences[user].append((item, timestamp, row_number))
    for sequence in sequences.values():
        sequence.sort(key=lambda row: (row[1], row[2]))
    sequences = {user: sequence for user, sequence in sequences.items()
                 if len(sequence) >= 3}

    # A held-out item without a training example would be cold-start.
    while sequences:
        trained_items = {item for seq in sequences.values() for item, _, _ in seq[:-2]}
        kept = {user: seq for user, seq in sequences.items()
                if seq[-2][0] in trained_items and seq[-1][0] in trained_items}
        if len(kept) == len(sequences):
            break
        sequences = kept
    if not sequences:
        raise ValueError('Filtering removed every user')

    user_map = {old: new for new, old in enumerate(sorted(sequences), 1)}
    item_map = {old: new for new, old in enumerate(
        sorted({item for seq in sequences.values() for item, _, _ in seq}), 1)}
    catalog = set(item_map.values())
    splits = {'train': [], 'dev': [], 'test': []}
    for old_user, seq in sorted(sequences.items()):
        user = user_map[old_user]
        clicked = {item_map[item] for item, _, _ in seq}
        pool = sorted(catalog - clicked)
        if len(pool) < negative_count:
            raise ValueError(f'User {old_user} lacks {negative_count} negatives')
        for item, timestamp, _ in seq[:-2]:
            splits['train'].append((user, item_map[item], timestamp))
        for offset, (phase, (item, timestamp, _)) in enumerate(
                zip(('dev', 'test'), seq[-2:])):
            negatives = random.Random(seed + user * 2 + offset).sample(pool, negative_count)
            splits[phase].append((user, item_map[item], timestamp, str(negatives)))

    output_dir.mkdir(parents=True, exist_ok=True)
    for phase, rows in splits.items():
        rows.sort(key=lambda row: (row[2], row[0], row[1]))
        with (output_dir / f'{phase}.csv').open('w', encoding='utf-8', newline='') as out:
            writer = csv.writer(out, delimiter='\t')
            writer.writerow(['user_id', 'item_id', 'time'] +
                            ([] if phase == 'train' else ['neg_items']))
            writer.writerows(rows)
    stats = {'users': len(user_map), 'items': len(item_map),
             **{phase: len(rows) for phase, rows in splits.items()}}
    (output_dir / 'preparation.json').write_text(
        json.dumps({'parameters': {'minimum_interactions': min_count,
                                   'negative_count': negative_count, 'seed': seed},
                    'statistics': stats}, indent=2), encoding='utf-8')
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-count', type=int, default=5)
    parser.add_argument('--negatives', type=int, default=99)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    print(prepare(args.ratings, args.output, args.min_count,
                  args.negatives, args.seed))
