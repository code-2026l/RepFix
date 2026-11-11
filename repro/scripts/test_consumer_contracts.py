import pytest
import torch
from consumer_contracts import confidence_thresholds,decision_flows,evaluate_contracts


def test_regressions_corrections_and_wrong_to_wrong_exactly_decompose_changes():
    r=decision_flows(torch.tensor([0,1,2,0,1]),torch.tensor([0,0,1,2,2]),torch.tensor([0,0,0,2,1]))
    assert (r['changed'],r['correct_to_wrong'],r['wrong_to_correct'],r['wrong_to_wrong'])==(4,1,2,1)
    assert r['negative_flip_rate']==.2 and r['new_correct']-r['old_correct']==1


def test_temperature_can_preserve_every_class_but_change_selective_consumer():
    old=torch.tensor([[4.,0.],[2.,0.],[.4,0.],[.1,0.]])
    thresholds=confidence_thresholds(old)
    rows=evaluate_contracts(old/10,old,torch.zeros(4,dtype=torch.long),thresholds)
    assert rows['classification']['disagreement']==0
    assert rows['coverage50']['disagreement']>0


def test_thresholds_are_training_only_and_are_not_mutated_by_evaluation():
    train=torch.tensor([[1.,0.],[3.,0.],[.1,0.]])
    thresholds=confidence_thresholds(train);before=dict(thresholds)
    evaluate_contracts(train*4,train,torch.zeros(3,dtype=torch.long),thresholds)
    assert thresholds==before


def test_empty_accepted_set_has_no_fabricated_accuracy():
    z=torch.zeros(5,2)
    row=evaluate_contracts(z,z,torch.zeros(5,dtype=torch.long),{'high':.99})['high']
    assert row['coverage']==0 and row['accepted_accuracy'] is None


def test_nonfinite_logits_fail_before_any_compatibility_claim():
    with pytest.raises(ValueError):confidence_thresholds(torch.tensor([[float('nan'),0.]]))
