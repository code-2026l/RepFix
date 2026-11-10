import numpy as np
from analyze_reference_replication import reference_interval


def test_uncertainty_uses_reference_means_not_twelve_continuations():
    baseline=np.array([.01,.02,.03,.04])[:,None]
    nested=baseline+np.array([-.1,0,.1])[None,:]
    grouped=reference_interval(nested);collapsed=reference_interval(np.repeat(baseline,3,axis=1))
    assert grouped['df']==3 and grouped['references']==4
    assert np.allclose(grouped['ci95'],collapsed['ci95'])
    assert np.allclose(grouped['reference_means'],baseline[:,0])


def test_degenerate_reference_means_have_finite_zero_width_interval():
    value=reference_interval(np.ones((4,3))*.02)
    assert value['ci95']==[.02,.02] and value['mean']==.02
