"""Independent local decision-count and accuracy replay of native CPU evidence."""
import argparse,hashlib,json
from pathlib import Path
import torch
from consumer_routing import decisions
from consumer_risk import certificate


def main(a):
    rows=[]
    for path in sorted(a.records.glob('n???.json')):
        record=json.loads(path.read_text());state=a.arrays/(path.stem+'_states')/'replay.pt'
        assert hashlib.sha256(state.read_bytes()).hexdigest()==record['state_sha256']
        raw=torch.load(state,map_location='cpu',weights_only=True);labels=raw['labels'];thresholds=list(raw['thresholds'].values())
        old_decision=decisions(raw['healthy_logits'],thresholds);checks=0
        for name,value in raw['served'].items():
            batch,role=name.split('_',1);expected=record['audits'][batch][role]
            accuracy=float((value['current_predictions']==labels).double().mean());assert accuracy==expected['own']['acc']
            changed=decisions(value['legacy_logits'],thresholds)!=old_decision
            for j,(key,contract) in enumerate(expected['contracts'].items()):
                assert int(changed[:,j].sum())==round(len(labels)*contract['disagreement'])
                bound=certificate(decisions(value['legacy_logits'],thresholds)[:,j].numpy(),old_decision[:,j].numpy(),raw['monitor'].numpy(),.05/(12*2*2*3))
                assert all(bound[k]==contract['certificate'][k] for k in bound);checks+=1
        for name in ['current','detached']:
            assert float((raw[name+'_logits'].argmax(1)==labels).double().mean())==record['native'][name]['acc']
            assert float(raw['pair_retrieval_hits'][name].double().mean())==record['native']['cached_prefix_pair_retrieval'][name]
        unprotected=decisions(raw['unprotected_logits'],thresholds)!=old_decision
        assert all(int(unprotected[:,j].sum())==round(len(labels)*c['disagreement']) for j,c in enumerate(record['native']['unprotected_contracts'].values()))
        rows.append(dict(endpoint=path.stem,consumer_checks=checks,state_sha256=record['state_sha256']))
    assert len(rows)==12
    result=dict(complete=True,endpoints=rows,total_consumer_checks=sum(r['consumer_checks'] for r in rows),all_decision_counts_exact=True,
        all_certificates_exact=True,all_current_and_detached_accuracies_exact=True,device='local_cpu',
        source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(dict(complete=True,endpoints=len(rows),consumer_checks=result['total_consumer_checks'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--arrays',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    torch.set_num_threads(2);main(p.parse_args())
