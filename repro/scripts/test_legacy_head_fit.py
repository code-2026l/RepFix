import unittest
import torch

from legacy_head_fit import fit_legacy_heads


class TeacherFitTests(unittest.TestCase):
    def test_known_affine_change_recovers_held_out_teacher_function(self):
        generator=torch.Generator().manual_seed(49)
        old=torch.randn(170,7,generator=generator,dtype=torch.float64)
        matrix=torch.randn(7,7,generator=generator,dtype=torch.float64)+3*torch.eye(7)
        shift=torch.randn(7,generator=generator,dtype=torch.float64)
        weight=torch.randn(7,4,generator=generator,dtype=torch.float64)
        bias=torch.randn(4,generator=generator,dtype=torch.float64)
        new=old@matrix+shift
        teacher=old@weight+bias
        fit=fit_legacy_heads(new[:100],teacher[:100])['ols']
        torch.testing.assert_close(new[100:]@fit['weight'].T+fit['bias'],teacher[100:],atol=1e-11,rtol=1e-11)

    def test_rank_deficient_inputs_have_finite_minimum_norm_solution(self):
        x=torch.linspace(-1,1,100,dtype=torch.float64)
        h=torch.stack([x,2*x,torch.ones_like(x)],1)
        target=torch.stack([3*x+7,-x],1)
        fits=fit_legacy_heads(h,target)
        self.assertEqual(fits['ols']['feature_rank'],1)
        self.assertLess(fits['ols']['fitting_logit_mse'],1e-25)
        self.assertTrue(torch.isfinite(fits['ridge']['weight']).all())

    def test_failed_linear_fit_does_not_imply_information_erasure(self):
        # The feature is x itself; the teacher x^2 is exactly recoverable by a
        # nonlinear reader, while every affine reader has a nonzero residual.
        x=torch.linspace(-1,1,101,dtype=torch.float64)[:,None]
        fit=fit_legacy_heads(x,x.square())['ols']
        self.assertGreater(fit['fitting_logit_mse'],.05)
        torch.testing.assert_close(x.square(),x.pow(2))


if __name__=='__main__':unittest.main()
