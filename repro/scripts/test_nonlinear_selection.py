"""Exact discrete validation decisions at the zero-benefit boundary."""
from fractions import Fraction

from select_nonlinear_configuration import exact_accuracy_mean


def test_exactly_tied_correct_counts_are_zero_not_a_positive_float_artifact():
    pairs = [(1/10,0/10,10),(2/10,0/10,10),(0/10,3/10,10)]
    assert sum(left-right for left,right,_ in pairs)>0
    assert exact_accuracy_mean(pairs)==0


def test_exact_sign_and_different_validation_denominators_are_respected():
    assert exact_accuracy_mean([(2/3,1/3,3),(3/4,1/4,4)])==Fraction(5,12)
    assert exact_accuracy_mean([(1/10,2/10,10),(3/10,3/10,10)])==Fraction(-1,20)
