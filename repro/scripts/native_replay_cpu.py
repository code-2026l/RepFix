"""Prepacked numerical C++ kernels and their independent folded comparator."""
import ctypes,hashlib,subprocess,time
from pathlib import Path
import numpy as np
import torch

FLAGS=['-std=c++17','-O3','-ffast-math','-march=native','-fPIC','-shared']

def compile_kernel(folder,d,c,s,r,sparse):
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True);source=Path(__file__).with_name('native_replay.cpp')
    binary=folder/('kernel_%d_%d_%d_%d_%d.so'%(d,c,s,r,sparse));start=time.perf_counter()
    argv=['g++',*FLAGS,'-DDIM='+str(d),'-DCLASSES='+str(c),'-DFIRST_RANK='+str(s),'-DSECOND_RANK='+str(r),'-DSPARSE='+str(int(sparse)),str(source),'-o',str(binary)]
    p=subprocess.run(argv,capture_output=True,timeout=120)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-2000:])
    return binary,dict(setup_seconds=time.perf_counter()-start,flags=FLAGS,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())

def pack(p,sparse=False):
    import dense_replay as dense
    import shared_reference_replay as supported
    merged=(supported if sparse else dense).merged_serving(p);m=64
    def transpose(t):return t.detach().float().T.contiguous().numpy()
    if sparse:
        s=len(p['support']);first=p['delta1'];bias=p['delta1_bias']
        left=np.zeros((1,),np.float32);bias1=np.zeros(m,np.float32);support=p['support'].float().numpy()
    else:
        s=p['q'].shape[0];first=torch.cat([p['first_weight'],p['q']]);bias=torch.cat([p['first_bias'],torch.zeros(s)])
        left=transpose(p['p']);bias1=p['delta1_bias'].numpy();support=np.zeros(1,np.float32)
    arrays=[transpose(merged['first_weight']),merged['first_bias'].numpy(),transpose(merged['second_weight']),merged['second_bias'].numpy(),
        transpose(p['current_weight']),p['current_bias'].numpy(),transpose(p['first_weight']),p['first_bias'].numpy(),
        transpose(p['second_weight']),p['second_bias'].numpy(),transpose(p['old_weight']),p['old_bias'].numpy(),
        transpose(first),bias.numpy(),left,bias1,transpose(p['v']),transpose(p['u']),p['delta2_bias'].numpy(),support]
    return [np.ascontiguousarray(v,dtype=np.float32) for v in arrays]

class NativeReplay:
    def __init__(self,binary,parameters,c):
        self.lib=ctypes.CDLL(str(Path(binary).resolve()));self.parameters=parameters;self.classes=c
        self.f=ctypes.POINTER(ctypes.c_float);self.pointers=(self.f*len(parameters))(*[a.ctypes.data_as(self.f) for a in parameters])
        self.operations={name:getattr(self.lib,'rf_'+name) for name in ['current','replay','independent','empty']}
        for op in self.operations.values():op.argtypes=[self.f,ctypes.POINTER(self.f),self.f,ctypes.c_int];op.restype=None
        self.lib.rf_repeat.argtypes=[self.f,ctypes.POINTER(self.f),self.f,ctypes.c_int,ctypes.c_int,ctypes.c_int];self.lib.rf_repeat.restype=None
    def call(self,x,mode='replay',out=None):
        x=np.ascontiguousarray(x,dtype=np.float32)
        if out is None:out=np.empty((len(x),self.classes*(2 if mode in ['replay','independent'] else 1)),dtype=np.float32)
        self.operations[mode](x.ctypes.data_as(self.f),self.pointers,out.ctypes.data_as(self.f),len(x));return out
    def timed_call(self,x,mode,out):
        xp=x.ctypes.data_as(self.f);yp=out.ctypes.data_as(self.f);op=self.operations[mode];n=len(x)
        return lambda:op(xp,self.pointers,yp,n)
    def repeat_call(self,x,mode,out,repetitions):
        code=['current','replay','independent','empty'].index(mode)
        xp=x.ctypes.data_as(self.f);yp=out.ctypes.data_as(self.f);n=len(x)
        return lambda:self.lib.rf_repeat(xp,self.pointers,yp,n,repetitions,code)
