from scripts.run_cpu_experiments import parse_log


def test_parse_framework_metrics_and_parameter_count():
    log = '''#params: 12345
    Best Iter(dev)=    3 dev=(NDCG@10:0.1111) [22.4 s]
    Test After Training: (HR@5:0.2000,NDCG@5:0.1200,HR@10:0.3500,NDCG@10:0.1700)
    '''
    parsed = parse_log(log)
    assert parsed['parameters'] == 12345
    assert parsed['best_epoch'] == 3
    assert parsed['metrics'] == {'HR@5': 0.2, 'NDCG@5': 0.12,
                                 'HR@10': 0.35, 'NDCG@10': 0.17}
