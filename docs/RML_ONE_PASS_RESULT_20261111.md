# Exact one-pass compatibility on RadioML

The fixed RadioML protocol uses width 64, 3% labeled modulation
signals and a 24-epoch joint or detached SNR update at rate 0.0001.
Development and confirmation seeds are disjoint, and confirmation
programs are frozen before loading test signals.

The first study found that nonlinear old-readout repair on frozen
current features could not reach the development risk budget.
The joint-initialized dual student retained a +0.32-point gain
[0.21, 0.42], but its worst old class disagreement was 5.68%.
Four of 16 references exceeded 5%. This program met both timing
caps.

A second study kept the healthy hidden units and added 16 ReLU
units to learn the current teacher output. The old affine head
has zero coefficients on those units. Both outputs use one
folded FP32 graph. The saved parameters confirm exact sharing
of the healthy subgraph. All 16 confirmation references have
zero old class and confidence decision changes on 4,400 test
signals per reference. Maximum FP32 logit error is 1.72e-5.

| Measure | Result |
| --- | --- |
| Accuracy gain over detached continuation | +0.33 points [0.15, 0.51] |
| Accuracy change from unrestricted joint | −0.05 points [−0.19, 0.09] |
| Isolated CPU latency versus current only, batch 1 | 1.016× [1.014, 1.017] |
| Isolated CPU latency versus current only, batch 128 | 1.070× [1.068, 1.072] |
| Isolated CPU latency versus two encoders, batch 128 | 0.545× [0.544, 0.546] |

The physical timing gate and the joint utility and compatibility
gate passed. The latency covers model forward on standardized
inputs. The reference is needed during offline training to supply
legacy targets and healthy hidden weights.

The final equal-work confirmation uses 16 fresh references. Both
exact/16 students have identical architecture, initialization,
data, label budget, update count and distillation work. Their
teachers differ in whether SNR gradients reached the encoder.

Joint-student accuracy exceeds detached-student accuracy by
0.28 points [0.11, 0.45]. The change from the unrestricted joint
teacher is −0.12 [−0.24, 0.002], meeting the fixed −0.25-point
noninferiority margin. Gains below 0 dB and at or above 0 dB are
0.30 [0.11, 0.49] and 0.27 [0.06, 0.48] points descriptively.
The healthy subgraph is verified in all 32 deployable programs.
Every audited old class and confidence decision agrees, with
maximum FP32 logit discrepancy 1.91e-5.

These results establish one-pass benefit, compatibility and the
specified CPU latency on this RadioML protocol. The same student
construction is not confirmed on a second task or backbone.
The retained units fix the old function, while the auxiliary gain
depends on distillation capacity and the data.
