import ast
import csv

from scripts.prepare_grocery_unique import prepare


def test_replaces_duplicate_candidates_without_changing_positives(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir()
    for phase, rows in {
        'train': [[1, 1, 100], [1, 2, 200], [2, 3, 100], [2, 4, 200],
                  [3, 5, 100], [3, 6, 200], [4, 7, 100], [4, 8, 200]],
        'dev': [[1, 3, 300, '[9, 9]'], [2, 5, 300, '[9, 9]'],
                [3, 7, 300, '[9, 9]'], [4, 1, 300, '[9, 9]']],
        'test': [[1, 4, 400, '[9, 9]'], [2, 6, 400, '[9, 9]'],
                 [3, 8, 400, '[9, 9]'], [4, 2, 400, '[9, 9]']],
    }.items():
        with (source / f'{phase}.csv').open('w', newline='', encoding='utf-8') as out:
            writer = csv.writer(out, delimiter='\t')
            writer.writerow(['user_id', 'item_id', 'time'] +
                            ([] if phase == 'train' else ['neg_items']))
            writer.writerows(rows)
    prepare(source, target, negative_count=2, seed=4)
    for phase in ('dev', 'test'):
        with (target / f'{phase}.csv').open(encoding='utf-8', newline='') as handle:
            table = list(csv.DictReader(handle, delimiter='\t'))
        assert len(table) == 4
        for row in table:
            assert len(set(ast.literal_eval(row['neg_items']))) == 2
            assert int(row['item_id']) not in ast.literal_eval(row['neg_items'])
