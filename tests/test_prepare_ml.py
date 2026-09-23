import csv
from pathlib import Path

from scripts.prepare_movielens_topk import prepare


def test_leave_two_out_and_negative_pool(tmp_path: Path):
    raw = tmp_path / 'ratings.dat'
    lines = []
    for user in range(1, 5):
        for step in range(6):
            item = ((2 * (user - 1) + step) % 8) + 1
            lines.append(f'{user}::{item}::5::{1000 + step * 100 + user}')
    raw.write_text('\n'.join(lines), encoding='utf-8')
    output = tmp_path / 'prepared'
    stats = prepare(raw, output, min_count=3, negative_count=2, seed=9)
    assert stats == {'users': 4, 'items': 8, 'train': 16, 'dev': 4, 'test': 4}
    tables = {}
    for split in ('train', 'dev', 'test'):
        with (output / f'{split}.csv').open(encoding='utf-8', newline='') as handle:
            tables[split] = list(csv.DictReader(handle, delimiter='\t'))
    for split in ('dev', 'test'):
        for row in tables[split]:
            negative = [int(x) for x in row['neg_items'].strip('[]').split(',')]
            clicked = {int(record['item_id']) for table in tables.values()
                       for record in table if record['user_id'] == row['user_id']}
            assert len(negative) == len(set(negative)) == 2
            assert not clicked.intersection(negative)
