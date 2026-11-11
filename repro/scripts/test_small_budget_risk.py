import torch
from small_budget_risk import forecast


def test_zero_residual_reproduces_a_well_separated_consumer():
    logits=torch.tensor([[5.,0.,-2.],[0.,4.,-3.]])
    assert forecast(logits,torch.zeros(32,3),[.7,.8],torch.randn(64,3,dtype=torch.double))==0


def test_common_logit_offset_does_not_change_forecast_decisions():
    torch.manual_seed(4);logits=torch.tensor([[4.,0.,-2.],[0.,4.,-2.]])
    residual=torch.randn(32,1).expand(-1,3)*10
    # Isotropic shrinkage invents transverse uncertainty for a common-offset
    # covariance. Center residuals and logits before statistical propagation.
    residual=residual-residual.mean(1,keepdim=True)
    assert forecast(logits,residual,[.6,.7],torch.randn(64,3,dtype=torch.double))==0
