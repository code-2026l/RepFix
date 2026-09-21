"""Exact minimum-change halfspace projection and checked finite SGD steps.

Guarantees refer to a fixed, explicitly evaluated representation batch, not to
unseen examples or population generalization. No momentum/Adam is hidden here.
"""
import torch


def project_update(proposal, variance_gradient, slack, rate=1.):
    if rate <= 0:
        raise ValueError('rate must be positive')
    norm2 = variance_gradient.square().sum()
    violation = torch.dot(variance_gradient, proposal) - rate * slack
    if float(norm2) == 0:
        if float(violation) > 0:
            raise ValueError('infeasible constraint with zero variance derivative')
        return proposal.clone()
    return proposal - violation.clamp_min(0) / norm2 * variance_gradient


def checked_step(parameters, direction, learning_rate, variance_fn, floor, max_backtracks=12):
    """Apply theta <- theta-eta*direction only if actual batch variance >= floor.

    Rejected steps restore all parameters. The caller owns optimizer state; this
    function uses a plain explicit update and no optimizer is advanced internally.
    """
    if learning_rate <= 0:
        raise ValueError('learning_rate must be positive')
    params = list(parameters)
    if sum(p.numel() for p in params) != direction.numel():
        raise ValueError('direction size mismatch')
    original = [p.detach().clone() for p in params]
    with torch.no_grad():
        initial = float(variance_fn())
        if initial < floor:
            raise ValueError('initial iterate is infeasible')
        try:
            for attempt in range(max_backtracks + 1):
                eta = learning_rate * 2.**(-attempt)
                offset = 0
                for p, old in zip(params,original):
                    p.copy_(old-eta*direction[offset:offset+p.numel()].view_as(p))
                    offset += p.numel()
                current = float(variance_fn())
                if torch.isfinite(torch.tensor(current)) and current >= floor:
                    return dict(accepted=True, eta=eta, backtracks=attempt, variance=current)
        except BaseException:
            for p,old in zip(params,original):p.copy_(old)
            raise
        for p,old in zip(params,original):p.copy_(old)
    return dict(accepted=False, eta=0., backtracks=max_backtracks+1, variance=initial)


def self_test():
    torch.manual_seed(17)
    dtype=torch.double
    g=torch.randn(30,dtype=dtype);v=torch.randn(30,dtype=dtype)
    for slack in (0.,.1,10.):
        u=project_update(g,v,slack)
        assert float(v@u) <= slack+1e-10
        # KKT: correction parallel to the constraint normal; zero if already feasible.
        delta=g-u
        torch.testing.assert_close(delta,v*((v@g-slack).clamp_min(0)/(v@v)))
    x=torch.randn(12,4,dtype=dtype)
    w=torch.nn.Parameter(torch.randn(4,3,dtype=dtype))
    def variance():
        h=x@w
        return (h-h.mean(0)).square().sum()/(2*len(x))
    val=variance();vg,=torch.autograd.grad(val,w)
    d=torch.randn_like(w)
    old=w.detach().clone();eps=1e-6
    with torch.no_grad():w.copy_(old-eps*d)
    measured=(float(variance())-float(val))/eps
    with torch.no_grad():w.copy_(old)
    assert abs(measured+float((vg*d).sum())) < 1e-4
    # An oversized contraction step must shrink its step size to preserve the floor.
    result=checked_step([w],old.flatten(),1.,variance,float(val)*.9)
    assert result['accepted'] and result['backtracks']>0
    assert float(variance())>=float(val)*.9
    frozen=w.detach().clone()
    def explode():
        if not torch.equal(w,frozen):raise RuntimeError('evaluation failed')
        return variance()
    try:checked_step([w],w.detach().flatten(),.1,explode,0.)
    except RuntimeError:pass
    else:raise AssertionError('test should raise')
    assert torch.equal(w,frozen)
    print('PASS: KKT optimum, parameter chain rule, finite-step backtracking and exception rollback')


if __name__=='__main__':self_test()
