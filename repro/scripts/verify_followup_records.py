"""Standard-library checks for the complete fixed-protocol follow-up release."""
import hashlib,json
from pathlib import Path


def verify(root):
    base=Path(root)/'repro/results/CPUCONT'
    read=lambda name:json.loads((base/name).read_text(encoding='utf-8'))
    routes=[json.loads(p.read_text()) for p in sorted((base/'conditional_reference_development').glob('p???.json'))]
    assert len(routes)==6 and sum(len(r['branches']) for r in routes)==480
    assert all(r['complete'] and not r['splits']['official_test_loaded'] and r['primary_model_not_modified'] for r in routes)
    route_choice=read('conditional_reference_choice.json')
    assert all(r['selected'] is None for r in route_choice['backbones'].values())
    for arch in ['resnet18','vit_b32']:
        own=[r for r in routes if r['arch']==arch];assert {r['seed'] for r in own}=={3800,3801,3802}
        assert route_choice['backbones'][arch]['lowest_referral_meeting_consumer_gate']>.5
    pilot=read('vision_auxiliary_first_pilot_choice.json');choice=read('vision_auxiliary_development_choice.json')
    assert pilot['complete'] and choice['complete']
    assert all(not r['extend_doses'] for r in pilot['backbones'].values())
    assert not choice['backbones']['resnet18']['confirm_ce_only']
    assert choice['backbones']['vit_b32']['confirm_ce_only']
    assert all(not r['confirm_auxiliary'] for r in choice['backbones'].values())
    development=[json.loads(p.read_text()) for p in sorted((base/'vision_auxiliary_development').glob('d???.json'))]
    assert len(development)==2 and all(r['complete'] and not r['splits']['official_test_loaded'] for r in development)
    confirmations=[json.loads(p.read_text()) for p in sorted((base/'vision_reference_confirmation').glob('c???.json'))]
    assert len(confirmations)==12
    assert {(r['reference_seed'],r['seed']) for r in confirmations}=={(r,s) for r in range(6201,6205) for s in range(6310,6313)}
    assert all(r['complete'] and r['phase']=='confirmation' and not r['splits']['official_test_loaded'] for r in confirmations)
    assert all({b['mode'] for b in r['branches']}=={'F','N','H','D'} for r in confirmations)
    summary=read('vision_reference_summary.json')
    assert summary['complete'] and not summary['auxiliary_benefit_confirmed'] and summary['shared_imagenet_pretraining']
    assert summary['reference_level_statistics']['H']['references']==4 and summary['reference_level_statistics']['H']['df']==3
    assert len({r['splits']['fitting_sha256'] for r in confirmations})==4
    assert len({r['splits']['validation_sha256'] for r in confirmations})==4
    assert len({r['splits']['mask_sha256'] for r in confirmations})==4
    for name,expected in summary['record_sha256'].items():
        assert hashlib.sha256((base/'vision_reference_confirmation'/name).read_bytes()).hexdigest()==expected
    replay=read('followup_output_replay.json')
    assert replay['complete'] and len(replay['routing'])==6 and len(replay['auxiliary'])==14
    assert sum(r['replayed_policies'] for r in replay['routing'])==480
    assert all(sum(r['old_decision_change_counts'])==0 for r in replay['auxiliary'])
    execution=read('followup_execution_audit.json')
    assert execution['complete'] and execution['all_owned_queues_stopped'] and not execution['gpu_compute_used']
    finite=read('followup_finite_state_audit.json')
    assert finite['complete'] and finite['all_returned_floating_tensors_finite'] and finite['states']==29
    assert finite['transport_audit_sha256']==hashlib.sha256((base/'followup_execution_audit.json').read_bytes()).hexdigest()
    return dict(routing_development_endpoints=6,routing_policies_replayed=480,
        real_visual_auxiliary_pilots=2,auxiliary_confirmation_runs=0,
        independent_fitted_references=4,transfer_confirmation_runs=12,
        saved_model_replays=14,primary_replication_unit='fitted reference',
        auxiliary_gain_confirmed=False,official_test_loaded=False)
