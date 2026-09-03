# Formal F10 step-50 / Slurm 136868

- Slurm result: `COMPLETED`, exit `0:0`, elapsed `02:31:28`, one physical node with two Pro 6000 GPUs
- Training result: completed optimizer steps `1-50`, saved one recoverable `global_step_50` checkpoint, and completed final validation
- Outcome signal: `21/50` steps (42%) had nonzero group-normalized outcome advantage and a finite nonzero gradient; `29/50` steps had all-zero outcome advantage
- Training reward: mean `0.1175` across 50 batches (`94/800` successful sampled trajectories); this is an online training statistic, not an independent benchmark result
- Optimization health: grad norm range `9.83e-6` to `0.12299`; effective-outcome-step grad norm range `0.00935` to `0.12299`; rollout/actor probability correlation mean `0.99911`; rollout-correction KL mean `0.00102`; clip fraction was zero throughout
- Validation: CAR dev mean@1 remained `0.269230769` from initial to final validation, so step 50 does not yet support a performance-improvement claim
- Runtime: mean/median step time `172.06/170.60 s`; mean throughput `1231.53 token/s`
- Telemetry peak memory: simulator `87,576/97,887 MiB` (89.47%); trainer `94,529/97,887 MiB` (96.57%); both reached 100% peak utilization
- Storage: checkpoint has 11 files and `31,443,788,637` bytes; it is the only full GRPO checkpoint, while SSD usage is `117.3/150 GB`
- Warning: a Ray DataLoader worker shutdown traceback appeared after progress reached 100%; checkpoint, final metrics, manifest, Slurm state, and exit code were all successful

Raw trainer, simulator, and role logs remain only on the controlled experiment server because they contain generated CAR conversations. Checkpoint tensors and serialized runtime state are also excluded from the public repository.
