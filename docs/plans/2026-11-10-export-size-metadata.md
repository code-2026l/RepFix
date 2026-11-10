# Descriptive storage cost for the already frozen serving choices

Before confirmation timing, also record the exact serialized TorchScript program
bytes of every already constructed current/protected implementation. Monolithic
programs count their whole export; modular programs count the shared prefix once
plus the suffix export. Serialize to memory after latency samples are collected,
without retaining another model file on the CPU host. This measures exported
program/constant storage, not process RSS, peak activation RAM or a theoretical
parameter lower bound. It cannot change any frozen ABI, training, precision,
decision threshold, utility/cost verdict or prior development outcome. The
numerical implementation and all manuscript work retain their existing limits.
