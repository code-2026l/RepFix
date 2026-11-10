"""Standard-library checks for the new frontier records in a source release."""
import hashlib,json,math,statistics

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(root,name):return json.loads((root/'repro/results/CPUCONT'/name).read_text(encoding='utf-8'))

def check_interval(row,values):
    assert len(values)==row['n']==10 and len(row['values'])==10
    for x,y in zip(values,row['values']):assert math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-14)
    mean=statistics.mean(values);half=2.2621571628540993*statistics.stdev(values)/math.sqrt(10)
    for x,y in zip([mean,mean-half,mean+half],[row['mean'],*row['ci95']]):assert math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-12)

def verify(root):
    base=root/'repro/results/CPUCONT'
    counts={};families={}
    for family in ['nonlinear_suffix_confirmation','two_block_confirmation']:
        rows=[json.loads(p.read_text()) for p in sorted((base/family).glob('c???.json'))]
        assert len(rows)==10 and {r['seed'] for r in rows}==set(range(3810,3820))
        assert all(r['complete'] and r['phase']=='confirmation' and r['splits']['official_test_loaded'] for r in rows)
        counts[family]=len(rows)
    choice=base/'reference_precision_choice.json'
    for family,timing in [('precision_native_confirmation',False),('precision_serving_confirmation',True)]:
        rows=[json.loads(p.read_text()) for p in sorted((base/family).glob('c???.json'))]
        assert len(rows)==20
        for arch in ['resnet18','vit_b32']:assert {r['seed'] for r in rows if r['arch']==arch}==set(range(3810,3820))
        assert all(r['complete'] and r['choice_sha256']==digest(choice) and r['timing_only']==timing and r['audit_only']!=timing for r in rows)
        families[family]=rows
        if not timing:
            for r in rows:
                for kind in ['chosen','full_precision']:
                    assert set(r['serving']['U'][kind])=={'1','128'}
                    for v in r['serving']['U'][kind].values():
                        assert set(v['contracts'])=={'classification','coverage80','coverage50'}
                        assert all(c['certificate']['monitor_examples']==1024 and c['certificate']['allocated_tail_probability']==.05/2000 for c in v['contracts'].values())
        counts[family]=len(rows)
    repairs=read(root,'suffix_repair_choice.json');assert not repairs['test_used_for_selection']
    assert set(repairs['backbones'])=={'resnet18','vit_b32'} and all(v['selected'] is None for v in repairs['backbones'].values())
    counts['affine_repair_development_arms']=2*3*9
    head=read(root,'vision_protected_head_choice.json')
    assert head['development_only'] and head['followup_after_unconstrained_test_opened'] and head['test_confirmation_runs']==0
    assert all(v['selected'] is None for v in head['backbones'].values())
    pilots=[json.loads(p.read_text()) for p in sorted((base/'vision_protected_head_pilot').glob('h???.json'))]
    assert len(pilots)==6 and all(r['complete'] and not r['splits']['official_test_loaded'] and not r['branches'][0]['passes'] for r in pilots)
    assert not (base/'vision_protected_head_confirmation').exists()
    initial=[json.loads(p.read_text()) for p in sorted((base/'vision_head_initialization').glob('i???.json'))]
    assert len(initial)==2 and all(r['complete'] for r in initial)
    assert all(v['disagreement']==0 for r in initial for v in r['branches'][1]['contracts'].values())
    counts.update(vision_distillation_pilots=6,vision_distillation_confirmations=0,teacher_initialization_audits=2)
    execution=read(root,'frontier_execution_audit.json');finite=read(root,'frontier_finite_state_audit.json');replay=read(root,'frontier_output_replay.json')
    assert execution['complete'] and execution['all_queues_stopped'] and not execution['gpu_compute_used']
    assert all(q['finished'] and not q['running'] for q in execution['queues'].values())
    assert execution['current_remote_result_bytes']<=16*2**30
    assert finite['complete'] and finite['all_floating_tensors_finite'] and finite['states']==execution['verified_state_files']
    assert finite['transport_audit_sha256']==digest(base/'frontier_execution_audit.json')
    expected={kind+'/'+name:v['sha256'] for kind,q in execution['queues'].items() for name,v in q['files'].items() if name.endswith(('.pt','.pth'))}
    assert finite['files_sha256']==expected
    assert replay['complete'] and replay['transport_audit_sha256']==digest(base/'frontier_execution_audit.json') and len(replay['endpoints'])==20
    assert all(len(r['checks'])==9 for r in replay['endpoints'])
    locations=read(root,'native_serving_record_locations.json');assert locations['complete'] and len(locations['records'])==20
    for r in replay['endpoints']:
        identifier='c%03d'%((r['seed']-3810)+(10 if r['arch']=='vit_b32' else 0))
        location=locations['records'][identifier];assert location['queue']==r['queue']
        assert location['record_sha256']==digest(base/'precision_native_confirmation'/(identifier+'.json'))
        assert expected[r['queue']+'/'+identifier+'_states/served_logits.pt']==r['state_sha256']
    resumed=execution['queues']['next_visionprecisionauditresume']
    assert resumed['memory_policy']['per_job_rss_bytes'] is None and resumed['memory_policy']['aggregate_rss_bytes'] is None
    assert resumed['memory_policy']['minimum_host_available_bytes']==256*2**30 and resumed['max_workers']<=4
    assert resumed['resource_plan_sha256']==digest(root/'docs/plans/2026-11-10-host-reserve-execution.md')
    counts.update(recovered_tensor_states=finite['states'],locally_replayed_output_groups=sum(len(r['checks']) for r in replay['endpoints']),
        local_output_replay_is_duplicate_backbone_execution=False,live_remote_result_bytes=execution['current_remote_result_bytes'],
        recovered_offloaded_bytes=execution['offloaded_bytes'])
    summary=read(root,'served_frontier_summary.json')
    assert summary['complete'] and set(summary['backbones'])=={'resnet18','vit_b32'}
    for arch,row in summary['backbones'].items():
        native=[r for r in families['precision_native_confirmation'] if r['arch']==arch]
        timing=[r for r in families['precision_serving_confirmation'] if r['arch']==arch]
        assert all(n['source_sha256']==t['source_sha256'] for n,t in zip(native,timing))
        for kind,u in row['utility'].items():
            assert all(r['serving']['U'][kind]['1']['own']['acc']==r['serving']['U'][kind]['128']['own']['acc'] for r in native)
            own=[r['serving']['U'][kind]['128']['own']['acc'] for r in native];check_interval(u['absolute'],own)
            for control,interval in u['differences'].items():
                values=[x-(r['healthy']['acc'] if control=='healthy' else r['audits'][control]['own']['acc']) for x,r in zip(own,native)]
                check_interval(interval,values)
        assert all(u['consumer_checks']==60 for u in row['utility'].values())
        for batch,cost in row['cost'].items():
            assert cost['budget']==(1.20 if batch=='1' else 1.10)
            for kind in ['chosen','full_precision']:
                selected=cost[kind]['backend'];current=cost['current_backend']
                values=[r['serving']['U']['timings'][batch]['medians_seconds'] for r in timing]
                check_interval(cost[kind]['ratio'],[v[selected]/v[current] for v in values])
                check_interval(cost[kind]['protected_seconds'],[v[selected] for v in values])
                assert cost[kind]['mean_within_budget']==(cost[kind]['ratio']['mean']<=cost['budget'])
                assert cost[kind]['upper_ci_within_budget']==(cost[kind]['ratio']['ci95'][1]<=cost['budget'])
    return counts
