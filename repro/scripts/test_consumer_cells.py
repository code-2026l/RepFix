import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from consumer_cells import cell_spec,project_cell,cell_loss,cell_logodds_loss


def test_teacher_is_feasible_and_has_zero_cell_loss():
    z=torch.tensor([[3.,1.,-1.],[.5,.4,.1],[1.,-.2,.1]],dtype=torch.float64)
    spec=cell_spec(z,[.4,.8]);assert cell_loss(z,spec).item()<1e-12
    p=project_cell(z.log_softmax(1),spec)
    assert torch.allclose(p,z.softmax(1),atol=1e-12)


def test_projection_matches_independent_constrained_optimizer():
    g=torch.Generator().manual_seed(91)
    old=torch.randn(12,5,generator=g,dtype=torch.float64)*1.5
    z=torch.randn(12,5,generator=g,dtype=torch.float64)*3
    spec=cell_spec(old,[.4,.75]);q=z.softmax(1);actual=project_cell(z.log_softmax(1),spec)
    for i in range(len(z)):
        k=int(spec['class'][i]);ratio=float(spec['ratio'][i]);lo=float(spec['lower'][i]);hi=float(spec['upper'][i])
        initial=old[i].softmax(0).numpy();logq=z[i].log_softmax(0).numpy()
        def objective(p):return float(np.sum(p*(np.log(p)-logq)))
        constraints=[{'type':'eq','fun':lambda p:p.sum()-1},
            {'type':'ineq','fun':lambda p,k=k,l=lo:p[k]-l},
            {'type':'ineq','fun':lambda p,k=k,u=hi:u-p[k]}]
        constraints += [{'type':'ineq','fun':lambda p,j=j,k=k,r=ratio:p[k]-r*p[j]} for j in range(5) if j!=k]
        result=minimize(objective,initial,method='SLSQP',bounds=[(1e-12,1)]*5,
            constraints=constraints,options={'ftol':1e-12,'maxiter':1000})
        assert result.success,result.message
        assert abs(objective(actual[i].numpy())-result.fun)<2e-8


def test_projection_retains_class_and_both_selective_consumers():
    old=torch.tensor([[2.,1.,0.],[3.,0.,0.],[.2,0.,.1]],dtype=torch.float64)
    q=torch.tensor([[-5.,5.,0.],[.1,0.,0.],[4.,0.,0.]],dtype=torch.float64)
    spec=cell_spec(old,[.45,.8]);p=project_cell(q.log_softmax(1),spec);ref=old.softmax(1)
    assert torch.equal(p.argmax(1),ref.argmax(1))
    for threshold in [.45,.8]:
        a=p.argmax(1);a[p.max(1).values<threshold]=-1
        b=ref.argmax(1);b[ref.max(1).values<threshold]=-1
        assert torch.equal(a,b)


def test_envelope_gradient_matches_independent_finite_difference():
    z=torch.tensor([[-1.,2.,.3]],dtype=torch.float64,requires_grad=True)
    spec=cell_spec(torch.tensor([[2.,0.,-.5]],dtype=torch.float64),[.5,.9])
    gradient=torch.autograd.grad(cell_loss(z,spec),z)[0];step=1e-5
    for j in range(3):
        plus=z.detach().clone();minus=z.detach().clone();plus[0,j]+=step;minus[0,j]-=step
        finite=(cell_loss(plus,spec)-cell_loss(minus,spec))/(2*step)
        assert abs(finite-gradient[0,j])<2e-7


def test_within_cell_distribution_changes_are_unpenalized():
    old=torch.tensor([[2.,0.,0.]],dtype=torch.float64)
    spec=cell_spec(old,[.5,.9]);new=torch.tensor([[2.2,.5,-.5]],dtype=torch.float64)
    assert cell_loss(new,spec)<1e-12
    assert torch.nn.functional.kl_div(new.log_softmax(1),old.softmax(1),reduction='batchmean')>.001


def test_shift_invariance_and_bounded_gradient_for_extreme_logits():
    old=torch.tensor([[2.,1.,0.],[3.,1.,-1.]],dtype=torch.float64)
    spec=cell_spec(old,[.5,.9]);z=torch.tensor([[-10000.,10000.,0.],[10000.,-10000.,-20000.]],dtype=torch.float64,requires_grad=True)
    loss=cell_loss(z,spec);gradient=torch.autograd.grad(loss,z)[0]
    assert torch.isfinite(loss) and torch.isfinite(gradient).all() and gradient.abs().max()<=.5+1e-12
    assert torch.allclose(loss,cell_loss(z.detach()+12345,spec),atol=1e-10)


def test_invalid_probability_interval_is_rejected():
    with pytest.raises(ValueError):cell_spec(torch.zeros(2,3),[1.1])


def test_single_class_and_nonfinite_teacher_are_rejected():
    with pytest.raises(ValueError):cell_spec(torch.ones(2,1),[.8])
    with pytest.raises(ValueError):cell_spec(torch.tensor([[float('nan'),0.]]),[.8])


def test_high_confidence_threshold_crossing_is_not_discounted_by_logodds_metric():
    values=[]
    for epsilon in [1e-3,1e-5,1e-7]:
        old=torch.tensor([[np.log(1-epsilon/2),np.log(epsilon/2)]],dtype=torch.float64)
        new=torch.tensor([[np.log(1-2*epsilon),np.log(2*epsilon)]],dtype=torch.float64)
        spec=cell_spec(old,[1-epsilon]);values.append((float(cell_loss(new,spec)),float(cell_logodds_loss(new,spec))))
    assert values[-1][0]<values[0][0]/1000
    assert values[-1][1]>.5 and abs(values[0][1]-values[-1][1])<.01


def test_logodds_loss_has_zero_and_offset_invariance_on_teacher_cell():
    old=torch.tensor([[2.,0.,1.],[.4,.3,.1]],dtype=torch.float64)
    spec=cell_spec(old,[.4,.8]);assert cell_logodds_loss(old,spec)<1e-20
    new=-old;assert torch.allclose(cell_logodds_loss(new,spec),cell_logodds_loss(new+999,spec),atol=1e-10)
