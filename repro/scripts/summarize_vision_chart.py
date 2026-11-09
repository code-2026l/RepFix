"""Local evaluation of the fixed one-seed visual development screen."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(path):
    record=json.loads(path.read_text())
    assert record['complete'] and record['phase']=='development' and record['seed']==900
    assert not record['splits']['official_test_loaded'] and not record['splits']['test_metrics_computed']
    assert [b['mode'] for b in record['branches']]==['D','J1','J10']
    assert len({b['minibatch_order_sha256'] for b in record['branches']})==1
    assert len({b['rotation_order_sha256'] for b in record['branches']})==1
    healthy=record['healthy']['acc']
    detached=record['branches'][0]
    result=dict(phase='development',seed=900,plan_sha256=record['plan_sha256'],
        record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),test_opened=False,
        healthy=record['healthy'],branches=[],promising=False,selected=None,
        selection_order='J1 then J10; budgets 128/512/1024/2048; diagonal then paired ridge')
    for branch in record['branches']:
        assert branch['bn_buffers_unchanged']
        loss=healthy-branch['frozen']['acc']
        item=dict(mode=branch['mode'],own=branch['own'],frozen=branch['frozen'],
            supervised_refit=branch['supervised_refit'],old_loss=loss,
            current_vs_detached=branch['own']['acc']-detached['own']['acc'],candidates=[])
        for budget in ['128','512','1024','2048']:
            for candidate in ['diagonal_stats','paired_ridge']:
                repaired=branch['repairs'][budget][candidate]
                fraction=(repaired['acc']-branch['frozen']['acc'])/loss if loss>0 else None
                checks=dict(healthy_quality=healthy>=.25,current_retained=branch['own']['acc']>=healthy-.005,
                    substantial_old_loss=loss>=.05,recovery_fraction=fraction is not None and fraction>=.8,
                    recovered_accuracy=repaired['acc']>=healthy-.01)
                passes=branch['mode']!='D' and all(checks.values())
                item['candidates'].append(dict(anchors=int(budget),candidate=candidate,metrics=repaired,
                                              recovery_fraction=fraction,checks=checks,passes=passes))
                if passes and result['selected'] is None:
                    result.update(promising=True,selected=dict(mode=branch['mode'],anchors=int(budget),candidate=candidate))
        result['branches'].append(item)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.record)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))
