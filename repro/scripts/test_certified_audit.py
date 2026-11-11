import itertools
import numpy as np
from certified_audit import count_interval,allocate,bounds,decide


def test_exact_hypergeometric_tails_cover_for_every_small_population_count():
    N=8;n=4;alpha=.05
    for K in range(N+1):
        bad=0;all_subsets=list(itertools.combinations(range(N),n))
        for sample in all_subsets:
            k=sum(i<K for i in sample);lo,hi=count_interval(N,n,k,alpha)
            bad+=not(lo<=K<=hi)
        assert bad/len(all_subsets)<=2*alpha


def test_census_recovers_all_consumer_risks_even_with_wrong_sampling_predictions():
    errors=np.array([[0,1],[1,0],[1,1],[0,0]])
    groups=[np.array([0,1]),np.array([2,3])];orders=[np.array([1,0]),np.array([0,1])]
    b=bounds(errors,groups,orders,[2,2],.001)
    assert b['lower']==b['upper']==[.5,.5]
    assert decide(b,.49)=='reject' and decide(b,.5)=='accept'


def test_sampling_allocation_respects_budget_floor_capacity_and_needs_no_outcomes():
    a=allocate([100,200,300,400],[.001,.01,.05,.2],128)
    assert sum(a)==128 and min(a)>=8 and np.all(a<=np.array([100,200,300,400]))
