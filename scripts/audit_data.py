"""Check a ReChorus Top-K dataset before running comparisons."""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def audit(directory: Path):
    data = {phase: pd.read_csv(directory / f'{phase}.csv', sep='\t')
            for phase in ('train', 'dev', 'test')}
    summary = {'dataset': directory.name, 'splits': {}}
    for phase, frame in data.items():
        if not {'user_id', 'item_id', 'time'}.issubset(frame.columns):
            raise ValueError(f'{phase}: missing ReChorus columns')
        if frame[['user_id', 'item_id']].min().min() < 1:
            raise ValueError(f'{phase}: zero or negative ID')
        if frame['time'].isna().any():
            raise ValueError(f'{phase}: missing timestamp')
        summary['splits'][phase] = {'rows': len(frame),
                                    'users': int(frame.user_id.nunique()),
                                    'items': int(frame.item_id.nunique()),
                                    'min_time': int(frame.time.min()),
                                    'max_time': int(frame.time.max())}
    all_data = pd.concat(data.values(), ignore_index=True)
    summary['users'] = int(all_data.user_id.nunique())
    summary['items'] = int(all_data.item_id.nunique())
    for column in ('user_id', 'item_id'):
        found = set(all_data[column])
        if found != set(range(1, max(found) + 1)):
            raise ValueError(f'{column}: IDs are not contiguous from 1')
    clicked = {int(user): set(frame.item_id) for user, frame in
               all_data.groupby('user_id')}
    summary['heldout_items_without_training'] = len(
        (set(data['dev'].item_id) | set(data['test'].item_id)) -
        set(data['train'].item_id))
    for phase in ('dev', 'test'):
        frame = data[phase]
        if 'neg_items' not in frame:
            raise ValueError(f'{phase}: no fixed negative candidates')
        if frame.user_id.duplicated().any():
            raise ValueError(f'{phase}: expected one target per user')
        lengths = set()
        duplicate_rows = clicked_rows = out_of_range = 0
        for user, raw in zip(frame.user_id, frame.neg_items):
            negatives = list(ast.literal_eval(raw))
            lengths.add(len(negatives))
            duplicate_rows += len(negatives) != len(set(negatives))
            clicked_rows += bool(clicked[int(user)].intersection(negatives))
            out_of_range += any(x < 1 or x > summary['items'] for x in negatives)
        summary['splits'][phase]['negative_counts'] = sorted(lengths)
        summary['splits'][phase]['duplicate_negative_rows'] = duplicate_rows
        summary['splits'][phase]['previously_clicked_negative_rows'] = clicked_rows
        summary['splits'][phase]['out_of_range_negative_rows'] = out_of_range
        if lengths != {99} or duplicate_rows or clicked_rows or out_of_range:
            raise ValueError(f'{phase}: invalid 99-negative candidate lists')
    train_latest = data['train'].groupby('user_id').time.max()
    dev_time = data['dev'].set_index('user_id').time
    test_time = data['test'].set_index('user_id').time
    comparable = train_latest.index.intersection(dev_time.index).intersection(test_time.index)
    summary['users_with_complete_sequence'] = len(comparable)
    summary['temporal_order_violations'] = int(
        ((train_latest.loc[comparable] > dev_time.loc[comparable]) |
         (dev_time.loc[comparable] > test_time.loc[comparable])).sum())
    if summary['temporal_order_violations']:
        raise ValueError('chronological train/dev/test order violated')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding='utf-8')
    print(encoded)
