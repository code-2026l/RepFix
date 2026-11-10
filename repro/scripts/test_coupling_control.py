import itertools
import numpy as np
from audit_coupling_control import extrema


def test_assignment_extrema_equal_exhaustive_permutations():
    cost=np.array([[0,1,1,0],[1,0,1,1],[0,0,1,1],[1,0,0,1]])
    lo,hi,a,b=extrema(cost)
    scores=[cost[np.arange(4),p].mean() for p in itertools.permutations(range(4))]
    assert lo==min(scores) and hi==max(scores)
    assert len(set(a))==len(set(b))==4
