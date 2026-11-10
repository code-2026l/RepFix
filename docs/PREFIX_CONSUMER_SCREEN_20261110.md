# Immutable-prefix consumer screen

The fixed screen tests a different computation access point: compress the old
consumer directly from the unchanged shared prefix, rather than updated terminal
features. Both backbones use reference6200, its original10000 fitting inputs
and5000 validation inputs. Training uses healthy logits, no consumer labels.

All eight candidates fail the1% maximum class/selective disagreement gate.
The best ResNet-18 candidate is pooled features plus the residual reader,
29.64%; the best ViT-B/32 candidate uses CLS/mean features plus the residual
reader,17.94%. These are development results, not lower bounds on what another
student could achieve. No confirmation or serving-cost evaluation follows.

Both CPU jobs finish normally; peak aggregate RSS is19.65GiB, with no fixed RSS
cap and the host-available reserve retained. Both state files, records and logs
are recovered with SHA-256. All eight decision arrays are replayed locally.
The submitted manuscript and its final packages remain unchanged. The new
research code/records are committed separately from the verified release.

This excludes one inexpensive shortcut under the specified recipe. It neither
establishes a new algorithm nor resolves novelty. The promising scientific
question remains when an update admits a cheap legacy interface, and whether
that property can be predicted before selecting an intervention. A serious
follow-up must compare equal teacher/data budgets and distinguish approximation
failure from actual loss of decodable information. Repeating larger students
until one passes would not establish that distinction.
