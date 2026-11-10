import itertools
import numpy as np
import pytest
import torch
from consumer_routing import decision_margin,decisions,oscillation,route_logits,oracle_referral

torch.set_num_threads(1)

def test_oscillation_margin_guarantees_all_named_consumers_and_ignores_offsets():
    g=torch.Generator().manual_seed(17)
    cheap=torch.randn(1000,6,generator=g,dtype=torch.double)*3
    reference=cheap+.13*torch.randn(1000,6,generator=g,dtype=torch.double)
    thresholds=[.6,.85]
    proven=decision_margin(cheap,thresholds)>oscillation(reference-cheap)
    assert proven.any() and torch.equal(decisions(cheap,thresholds)[proven],decisions(reference,thresholds)[proven])
    shifted=reference+torch.linspace(-100,100,len(reference))[:,None]
    assert torch.allclose(oscillation(reference-cheap),oscillation(shifted-cheap),atol=1e-12)

def test_confident_class_agreement_can_still_cross_a_legacy_threshold():
    a=torch.tensor([[6.,0.]],dtype=torch.double);b=torch.tensor([[4.,0.]],dtype=torch.double)
    assert decisions(a,[.99])[0,0]==decisions(b,[.99])[0,0]
    assert decisions(a,[.99])[0,1]!=decisions(b,[.99])[0,1]
    assert decision_margin(a,[.99])[0]<oscillation(a-b)[0]

def test_reference_fallback_only_removes_existing_disagreements():
    a=torch.tensor([[2.,0.],[0.,2.],[1.,0.]])
    b=torch.tensor([[0.,3.],[3.,0.],[1.,0.]])
    mask=torch.tensor([True,False,True])
    routed=route_logits(a,b,mask)
    assert torch.equal(routed[mask],b[mask]) and torch.equal(routed[~mask],a[~mask])
    before=decisions(a,[.8])!=decisions(b,[.8]);after=decisions(routed,[.8])!=decisions(b,[.8])
    assert torch.all(~after | before) and not after[mask].any()

def test_error_overlap_changes_the_oracle_mass_at_equal_marginal_errors():
    shared=np.zeros((10,3),bool);shared[:4]=True
    disjoint=np.zeros((10,3),bool);disjoint[:4,0]=True;disjoint[4:8,1]=True;disjoint[[0,1,8,9],2]=True
    a=oracle_referral(shared,.1);b=oracle_referral(disjoint,.1)
    assert a['minimum_fraction']==pytest.approx(.3)
    assert b['minimum_fraction']>a['minimum_fraction']

def test_fractional_oracle_is_a_lower_bound_for_every_integral_route():
    e=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1],[0,0,0]],dtype=bool)
    best=1.
    for bits in itertools.product([False,True],repeat=len(e)):
        selected=np.asarray(bits)
        if np.all((e & ~selected[:,None]).mean(0)<=.2+1e-12):best=min(best,float(selected.mean()))
    assert oracle_referral(e,.2)['minimum_fraction']<=best+1e-12
    assert oracle_referral(e,0)['minimum_fraction']==pytest.approx(.8)

def test_invalid_or_nonfinite_decision_inputs_are_rejected():
    with pytest.raises(ValueError):decision_margin(torch.tensor([[float('nan'),0.]]),[.8])
    with pytest.raises(ValueError):decisions(torch.ones(4,1),[.8])
    with pytest.raises(ValueError):decisions(torch.ones(4,2),[1.1])
