# Recorded source versions

Experiment drivers can evolve after an earlier run. The subdirectories retain
the exact bytes needed to check earlier records without changing those records
or replacing the current implementation.

The directory name is the digest recorded by the run. Most records use raw-file
SHA-256. Historical `vision_scale_replay.py` records used
`paired_readout_continuation.state_hash(bytes)`, which computes SHA-256 of
`repr(bytes).encode()`; their `vision_boundary_cpu.py` entry uses the same
convention. The release manifest always hashes the actual file bytes.
`verify_supplement.py` checks both conventions only for these two historical
vision filenames and reports their counts separately.

All archived versions match their recorded digest exactly. The source-recovery
process did not alter numerical results. Current entry points remain under
`repro/scripts/`; these snapshots preserve provenance rather than introduce
additional experimental choices.
