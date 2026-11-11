import copy
from summarize_vision_auxiliary import first_pilot


def row():
    branches=[]
    for name,acc in [('F',.5),('N',.5),('H',.5),('D',.502),('J0.01',.504),('J0.1',.5038),('J1',.504)]:
        branch=dict(mode=name,validation=dict(own=dict(acc=acc),reconstruction_mse=.95,
            old_decisions_unchanged=True),warm_optimizer_sha256='same')
        if name.startswith('J'):branch['alpha']=float(name[1:])
        if name=='D':branch['detached_mse_by_dose']={'0.01':1.,'0.1':1.,'1.0':1.}
        branches.append(branch)
    return dict(complete=True,phase='development',reference_seed=6200,seed=6300,
        splits=dict(official_test_loaded=False,validation=5000),branches=branches)


def test_gate_uses_integer_counts_at_exact_point_two_boundary():
    result=first_pilot(row())
    assert result['extend_doses']==[.01,1.]
    assert not result['confirm_ce_only_if_no_auxiliary']


def test_gate_requires_dose_matched_decoder_and_all_conditions():
    value=row();value['branches'][3]['detached_mse_by_dose']['0.01']=.98
    value['branches'][-1]['validation']['old_decisions_unchanged']=False
    result=first_pilot(value)
    assert result['extend_doses']==[] and result['confirm_ce_only_if_no_auxiliary']


def test_incomplete_or_mismatched_warm_state_cannot_select():
    value=row();value['branches'][-1]['warm_optimizer_sha256']='different'
    try:first_pilot(value)
    except AssertionError:pass
    else:raise AssertionError('Aliased or unpaired pilot accepted')
