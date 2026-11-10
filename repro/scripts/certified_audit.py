"""Fixed-population stratified audit; predictions affect sampling, not validity."""
import numpy as np
from consumer_risk import upper_total_disagreements


def count_interval(N,n,k,alpha):
    if not 0<n<=N:raise ValueError('Nonempty samples inside population required')
    if n==N:return int(k),int(k)
    return N-upper_total_disagreements(int(N),int(n),int(n-k),float(alpha)),upper_total_disagreements(int(N),int(n),int(k),float(alpha))


def allocate(sizes,scores,budget):
    sizes=np.asarray(sizes,dtype=int);scores=np.asarray(scores,dtype=float)
    if budget<8*len(sizes) or budget>sum(sizes) or min(sizes)<8:raise ValueError('Budget or stratum is too small')
    weights=.2*sizes/sum(sizes)+.8*(sizes*np.sqrt(np.maximum(scores,1e-8)))/sum(sizes*np.sqrt(np.maximum(scores,1e-8)))
    counts=np.minimum(sizes,8)
    while counts.sum()<budget:
        priority=weights/(counts+1);priority[counts>=sizes]=-np.inf;counts[np.argmax(priority)]+=1
    return counts


def bounds(errors,groups,orders,counts,alpha):
    errors=np.asarray(errors,dtype=int);N=len(errors);lo=np.zeros(errors.shape[1]);hi=lo.copy();estimate=lo.copy()
    for group,order,n in zip(groups,orders,counts):
        obs=errors[group[order[:n]]].sum(0);size=len(group);estimate+=size/N*obs/n
        for j,k in enumerate(obs):
            lower,upper=count_interval(size,int(n),int(k),alpha);lo[j]+=lower/N;hi[j]+=upper/N
    return dict(lower=lo.tolist(),upper=hi.tolist(),estimate=estimate.tolist())


def decide(bound,budget):
    if max(bound['lower'])>budget:return 'reject'
    if max(bound['upper'])<=budget:return 'accept'
    return 'abstain'
