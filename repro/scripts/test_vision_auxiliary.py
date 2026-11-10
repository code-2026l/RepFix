import copy
import torch
from torch import nn
from vision_auxiliary_cpu import primary_loss,augment,clone_decoder

torch.set_num_threads(1)

def test_unlabeled_batch_has_finite_loss_without_reading_its_hidden_labels():
    head=nn.Linear(4,3);h=torch.randn(16,4,requires_grad=True);empty=torch.zeros(16,dtype=torch.bool)
    a=primary_loss(head,h,torch.zeros(16,dtype=torch.long),empty)
    b=primary_loss(head,h,torch.full((16,),99,dtype=torch.long),empty)
    assert torch.isfinite(a) and torch.equal(a,b)
    a.backward();assert h.grad is not None and not h.grad.any()

def test_detached_reconstruction_cannot_modify_primary_parameter_gradient():
    encoder=nn.Linear(4,4);head=nn.Linear(4,3);decoder=nn.Linear(4,8);x=torch.randn(12,4);y=torch.randint(3,(12,));target=torch.randn(12,8)
    h=encoder(x);base=primary_loss(head,h,y,torch.ones(12,dtype=torch.bool))
    first=torch.autograd.grad(base,tuple(encoder.parameters()),retain_graph=True)
    joint=base+(decoder(h.detach())-target).square().mean()
    second=torch.autograd.grad(joint,tuple(encoder.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(first,second))

def test_cloning_decoder_optimizer_preserves_every_warm_moment():
    base=nn.Linear(4,3);opt=torch.optim.AdamW(base.parameters(),lr=.001)
    opt.zero_grad();base(torch.randn(8,4)).square().mean().backward();opt.step()
    snapshot=copy.deepcopy(opt.state_dict());clone,other=clone_decoder(base,opt)
    for a,b in zip(opt.state.values(),other.state.values()):
        assert all(torch.equal(a[k],b[k]) for k in ['step','exp_avg','exp_avg_sq'])
        assert all(a[k].data_ptr()!=b[k].data_ptr() for k in ['step','exp_avg','exp_avg_sq'])
    other.zero_grad();clone(torch.randn(8,4)).square().mean().backward();other.step()
    for before,after in zip(snapshot['state'].values(),opt.state.values()):
        assert all(torch.equal(before[k],after[k]) for k in ['step','exp_avg','exp_avg_sq'])

def test_augmentation_is_fixed_and_does_not_change_image_count_or_dtype():
    x=torch.randint(256,(12,32,32,3),dtype=torch.uint8)
    a,ha=augment(x);torch.manual_seed(100);b,hb=augment(x)
    assert torch.equal(a,b) and ha==hb and a.shape==x.shape and a.dtype==x.dtype
