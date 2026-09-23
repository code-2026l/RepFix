"""Numerical checks of the new diagnostic identities, not training evidence."""
import json
from pathlib import Path
import numpy as np

rng=np.random.default_rng(20260923)
errors=[];certified=0
for trial in range(3000):
    B,m,p=5,3,8
    C=np.kron(np.eye(B)-np.ones((B,B))/B,np.eye(m))
    J=rng.normal(size=(B*m,p));h=rng.normal(size=B*m);u=C@h
    gp=rng.normal(size=B*m);ga=rng.normal(size=B*m)
    K=J@J.T;k=np.trace(K)/len(h);E=K/k-np.eye(len(h))
    e=u/np.linalg.norm(u);np_=np.linalg.norm(gp)
    a=e@ga/np_;d=-e@gp/np_;ea=e@E@ga/np_;ep=-e@E@gp/np_
    alpha=10.**rng.uniform(-2,2);q=10.**rng.uniform(-2,2)
    N=k*np.linalg.norm(u)*np_/B
    exact=-u@K@(gp+alpha*ga)/B
    discrepancy=(d-1)+ep-alpha*(a-q+ea)
    reconstructed=N*(1-alpha*q+discrepancy)
    errors.append(abs(exact-reconstructed)/(1+abs(exact)))
    np.testing.assert_allclose(exact,reconstructed,rtol=1e-11,atol=1e-10)
    envelope=abs(d-1)+abs(ep)+alpha*(abs(a-q)+abs(ea))
    if abs(1-alpha*q)>envelope:
        certified+=1
        assert np.sign(exact)==np.sign(1-alpha*q)
    # For a linear encoder the exact finite-step remainder is quadratic.
    step=rng.normal(size=p)*.05
    v=J.T@u/B
    difference=(np.linalg.norm(C@(h+J@step))**2-np.linalg.norm(u)**2)/(2*B)
    remainder=difference-v@step
    hessian=J.T@C@J/B;L=np.linalg.eigvalsh(hessian)[-1]
    assert abs(remainder)<=L*np.linalg.norm(step)**2/2+1e-12
    np.testing.assert_allclose(remainder,step@hessian@step/2,atol=1e-12)

# Realizable quadratic-loss counterexample in the appendix.
beta=2.;v=np.array([-1.,beta]);gp=np.array([1.,0.]);ap=np.array([0.,1.]);am=-ap
np.testing.assert_equal(np.stack([gp,ap])@np.stack([gp,ap]).T,
                        np.stack([gp,am])@np.stack([gp,am]).T)
assert -v@(gp+ap)<0 < -v@(gp+am)
# Restoration-misalignment false negative; sign-loss false positive.
assert 1-.2>0 and .1-.2<0
assert 1-2<0 and 1-(-2)>0

# Non-vacuous sufficient certificates near the radial regime.
radial_certified=0
for _ in range(500):
    n=10;e=rng.normal(size=n);e/=np.linalg.norm(e)
    gp=-e;ga=2*e;alpha=float(rng.choice([.1,1,10]));q=2.
    A=rng.normal(size=(n,n));K=np.eye(n)+.001*A@A.T;E=K-np.eye(n)
    a=e@ga;d=-e@gp;ea=e@E@ga;ep=-e@E@gp
    Delta=abs(d-1)+abs(ep)+alpha*(abs(a-q)+abs(ea))
    assert abs(1-alpha*q)>Delta
    assert np.sign(-e@K@(gp+alpha*ga))==np.sign(1-alpha*q)
    radial_certified+=1
report=dict(random_checks=3000,certified_random=certified,radial_certificates=radial_certified,
            max_relative_identity_error=max(errors),finite_step_checks=3000,
            conflict_counterexample='passed',directional_error_examples='passed')
out=Path(__file__).resolve().parent/'qa/theory_checks.json'
out.write_text(json.dumps(report,indent=2),newline='\n')
print(json.dumps(report,indent=2))
