import cross_domain_mtl as C
D = C.load_domain('battery')
print('battery train', D['X_tr'].shape, 'test', D['X_te'].shape, 'n_cells', D.get('n_cells'))
recs, agg = C.collect_domain('battery', ['joint', 'stf0', 'stfsoft'], [0, 1], 30000.0, 64, 20, 1e-3, 256, 0.02, 8.0)
for a in agg:
    print(a['mode'], 'metric', round(a['metric_mean'], 4),
          'collapse', a['collapse_rate'], 'recovery', a['recovery_rate'])