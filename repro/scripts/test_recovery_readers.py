import pytest
import torch
from recovery_readers import RecoveryReader,export_reader,reader_logits,pack_outputs,serve_outputs


@pytest.mark.parametrize('architecture',['affine','quadratic','residual32'])
def test_served_map_and_packed_current_output_match_unfolded_reader(architecture):
    torch.manual_seed(331);width,classes=7,4
    mu=torch.randn(1,width,dtype=torch.float64);sd=torch.rand(1,width,dtype=torch.float64)+.5
    initial=dict(weight=torch.randn(classes,width,dtype=torch.float64),bias=torch.randn(classes,dtype=torch.float64))
    reader=RecoveryReader(width,classes,architecture,mu,sd,initial)
    with torch.no_grad():
        if architecture=='quadratic':reader.quadratic.weight.normal_(0,.2)
        if architecture=='residual32':reader.output.weight.normal_(0,.2)
        h=torch.randn(31,width);native=export_reader(reader);current=torch.nn.Linear(width,5)
        now,old=serve_outputs(h,pack_outputs(current,native))
        assert torch.allclose(reader(h),reader_logits(h,native),atol=6e-6)
        assert torch.allclose(old,reader(h),atol=6e-6)
        assert torch.allclose(now,current(h),atol=1e-6)
