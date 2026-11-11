"""Regression checks for scripted routing on empty, full and mixed selections."""
import pytest
import torch
from torch import nn

from vit_scripted_deployment_cpu import compile_current_tail
from vit_student_native_cpu import trace
from suffix_serving import WholeNetwork


class Student(nn.Module):
    def forward(self,x):
        return x[:,:3],2*x[:,:2]


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('shift',torch.tensor([.5,-.5,0.]))

    def forward(self,x):
        return x[:,:3]+self.shift


@pytest.mark.parametrize('indices',[(0,),(1,),(0,1)])
def test_scripted_route_preserves_current_and_old_outputs(indices):
    sample=torch.tensor([[1.,.95,0.,0.],[3.,0.,0.,0.]])
    program=compile_current_tail(Student(),Teacher(),.1,sample,freeze=False)
    program=trace(WholeNetwork(nn.Identity(),program).eval(),sample[:1])
    assert 'prim::If' in str(program.inlined_graph)
    x=sample[list(indices)]
    current,old=program(x)
    expected=x[:,:3].clone()
    top=expected.topk(2,dim=1).values
    selected=(top[:,0]-top[:,1])<=.1
    expected[selected]+=torch.tensor([.5,-.5,0.])
    torch.testing.assert_close(current,expected,atol=0,rtol=0)
    torch.testing.assert_close(old,2*x[:,:2],atol=0,rtol=0)
