"""Execute a fixed current-tail route with a scripted empty-set branch."""
import argparse
import json
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from consumer_routing import decisions
from suffix_serving import WholeNetwork
from vit_nonlinear_student_cpu import NonlinearVisualStudent,base,factory
from vit_student_native_cpu import TeacherSuffix,trace


class ScriptedCurrentTail(nn.Module):
    def __init__(self,student,teacher,cutoff):
        super().__init__()
        self.student=student
        self.teacher=teacher
        self.cutoff=cutoff

    def forward(self,tokens):
        z,old=self.student(tokens)
        top=z.topk(2,dim=1)[0]
        indices=((top[:,0]-top[:,1])<=self.cutoff).nonzero().flatten()
        if indices.numel()>0:
            replacement=self.teacher(tokens.index_select(0,indices))
            z=z.index_copy(0,indices,replacement)
        return z,old


class PrimaryOutput(nn.Module):
    def __init__(self,program):
        super().__init__()
        self.program=program

    def forward(self,x):
        return self.program(x)[0]


def pin_physical_cores():
    allowed=sorted(os.sched_getaffinity(0))
    physical={}
    for cpu in allowed:
        root=Path(f'/sys/devices/system/cpu/cpu{cpu}')
        package=int((root/'topology/physical_package_id').read_text())
        core=int((root/'topology/core_id').read_text())
        nodes=list(root.glob('node[0-9]*'))
        node=int(nodes[0].name[4:]) if nodes else package
        physical.setdefault((node,package,core),cpu)
    nodes=sorted({key[0] for key in physical})
    candidates=[[cpu for key,cpu in physical.items() if key[0]==node] for node in nodes]
    chosen=next(values[:12] for values in candidates if len(values)>=12)
    for thread in Path('/proc/self/task').iterdir():
        os.sched_setaffinity(int(thread.name),chosen)
    return chosen


@torch.no_grad()
def run(args):
    cache=torch.load(args.training_population,map_location='cpu',mmap=True,weights_only=True)
    population=torch.load(args.population,map_location='cpu',mmap=True,weights_only=True)
    record=json.loads(args.student_record.read_text())
    source=args.student_record.parent/(args.student_record.stem+'_states/program.pt')
    assert base.digest(source)==record['program_sha256']
    state=torch.load(source,map_location='cpu',weights_only=True)
    previous_record=json.loads(args.prior_audit_record.read_text())
    previous_path=args.prior_audit_record.parent/(args.prior_audit_record.stem+'_states/replay.pt')
    assert base.digest(previous_path)==previous_record['replay_sha256']
    previous=torch.load(previous_path,map_location='cpu',weights_only=True)
    assert previous_record['student_program_sha256']==record['program_sha256']
    rescue=json.loads(args.rescue_choice.read_text())
    ref=state['reference']
    prefix,healthy=factory.build_for(cache)
    healthy.load_state_dict(cache['healthy'])
    healthy.eval().requires_grad_(False)
    teacher_record=json.loads(args.teacher_record.read_text())
    endpoint=args.teacher_record.parent/(args.teacher_record.stem+'_states/endpoint.pt')
    assert base.digest(endpoint)==state['records']['joint']==teacher_record['state_sha256']
    teacher_state=torch.load(endpoint,map_location='cpu',weights_only=True)
    _,teacher=factory.build_for(cache)
    teacher.load_state_dict(teacher_state['model'])
    teacher.eval().requires_grad_(False)
    legacy=dict(weight=torch.cat((ref['heads']['coarse']['weight'],ref['heads']['fine']['weight'])),
                bias=torch.cat((ref['heads']['coarse']['bias'],ref['heads']['fine']['bias'])))
    student=NonlinearVisualStudent(healthy,state['models']['joint/128'],legacy).eval().requires_grad_(False)
    fitting=torch.cat((cache['h'][0,ref['fitting']],cache['hidden'][0,ref['fitting']]),1)
    z=torch.cat([student.readout(batch) for batch in fitting.split(256)])
    top=z.topk(2,dim=1).values
    cutoff=torch.quantile(top[:,0]-top[:,1],rescue['percentile']).item()
    assert cutoff==previous['cutoffs']['joint']
    sample_tokens=population['tokens'][:128]
    student_jit=trace(student,sample_tokens[:1])
    teacher_tail=TeacherSuffix(teacher,teacher_state['reader']).eval()
    teacher_jit=trace(teacher_tail,sample_tokens[:1])
    program=torch.jit.freeze(torch.jit.script(ScriptedCurrentTail(student_jit,teacher_jit,cutoff).eval()))
    assert 'prim::If' in str(program.inlined_graph)
    indices=torch.arange(0,128,4)
    output={}
    if args.stage=='audit':
        current,old=[],[]
        for tokens in population['tokens'].split(128):
            z,legacy_z=program(tokens)
            current.append(z.argmax(1))
            old.append(legacy_z)
        current,old=torch.cat(current),torch.cat(old)
        expected=torch.cat((previous['old'],previous['fine_old']),1)
        cuts=(previous['coarse_thresholds'],previous['fine_thresholds'])
        reference_rules=torch.cat((decisions(expected[:,:20],cuts[0]),decisions(expected[:,20:],cuts[1])),1)
        served_rules=torch.cat((decisions(old[:,:20],cuts[0]),decisions(old[:,20:],cuts[1])),1)
        single_current,single_old=[],[]
        for index in indices:
            z,legacy_z=program(sample_tokens[index:index+1])
            single_current.append(z.argmax(1))
            single_old.append(legacy_z)
        single_current,single_old=torch.cat(single_current),torch.cat(single_old)
        single_rules=torch.cat((decisions(single_old[:,:20],cuts[0]),decisions(single_old[:,20:],cuts[1])),1)
        output=dict(current=current,previous_current=previous['compiled_current'],reference_rules=reference_rules,
                    served_rules=served_rules,indices=indices,single_current=single_current,single_rules=single_rules,
                    labels=population['labels'],
                    current_changes=int((current!=previous['compiled_current']).sum()),
                    old_changes=(served_rules!=reference_rules).sum(0).tolist(),
                    single_current_changes=int((single_current!=current[indices]).sum()),
                    single_old_changes=(single_rules!=reference_rules[indices]).sum(0).tolist())
    else:
        images,_,_,sha=base.late.images(False)
        assert sha==population['data_sha256']
        sample=base.late.preprocess(images[:128],cache['metadata']['image_size'])
        only=PrimaryOutput(program).eval()
        raw=dict(teacher=WholeNetwork(prefix,teacher_tail).eval(),
                 student=WholeNetwork(prefix,PrimaryOutput(student)).eval(),
                 protected=WholeNetwork(prefix,program).eval(),
                 rescue_current=WholeNetwork(prefix,only).eval())
        compiled={name:trace(module,sample[:1]) for name,module in raw.items()}
        timings={}
        for batch in (1,128):
            for fn in compiled.values():
                for _ in range(6 if batch==1 else 3):
                    fn(sample[:batch])
            times={name:[] for name in compiled}
            for block in range(9):
                names=list(compiled)
                if block%2:
                    names.reverse()
                for name in names:
                    start=time.perf_counter_ns()
                    if batch==1:
                        for index in indices:
                            compiled[name](sample[index:index+1])
                        duration=(time.perf_counter_ns()-start)/len(indices)
                    else:
                        for _ in range(2):
                            compiled[name](sample)
                        duration=(time.perf_counter_ns()-start)/2
                    times[name].append(duration)
            timings[str(batch)]=times
        check=compiled['protected'](sample)
        pure=compiled['rescue_current'](sample)
        output=dict(timings=timings,current_changes=int((check[0].argmax(1)!=pure.argmax(1)).sum()),
                    indices=indices,
                    old_changed=(decisions(check[1][:,:20],previous['coarse_thresholds'])!=
                                 decisions(previous['old'][:128],previous['coarse_thresholds'])).sum(0).tolist(),
                    fine_old_changed=(decisions(check[1][:,20:],previous['fine_thresholds'])!=
                                      decisions(previous['fine_old'][:128],previous['fine_thresholds'])).sum(0).tolist())
    folder=args.out.parent/(args.out.stem+'_states')
    folder.mkdir(exist_ok=True,parents=True)
    replay=folder/'replay.pt'
    torch.save(output,replay)
    base.save(args.out,dict(complete=True,device='cpu',stage=args.stage,seed=record['seed'],
        reference_seed=record['reference_seed'],plan_sha256=args.plan_sha256,
        rescue_choice_sha256=base.digest(args.rescue_choice),student_program_sha256=record['program_sha256'],
        previous_replay_sha256=previous_record['replay_sha256'],replay_sha256=base.digest(replay),
        official_test_loaded=True,affinity=args.affinity,threads=torch.get_num_threads(),
        source_sha256={Path(__file__).name:base.digest(Path(__file__))},branches=[]))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('audit','timing'),required=True)
    for name in ('training-population','population','student-record','prior-audit-record','teacher-record','rescue-choice'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.affinity=pin_physical_cores() if args.stage=='timing' else sorted(os.sched_getaffinity(0))
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    run(args)
