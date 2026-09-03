# F10 step-6 resume / Slurm 136347

- Transfer archive SHA-256: `8c939ac3314cf62d1bb8a09d5b2eb33dedffd036b8e0c4b01c1b1919371f03ec`
- Slurm result: `COMPLETED`, exit `0:0`, elapsed `00:11:10`, node `gpu-pro6000-10`
- Resume result: loaded `global_step_5` with model/optimizer/extra state and completed optimizer step 6 plus final validation
- Storage result: no `global_step_6`; the original step-5 checkpoint remained the only full GRPO checkpoint
- Step-6 signal: outcome reward and advantage were all zero; the finite `1.17578e-5` gradient was KL-only, so this step is resume evidence rather than outcome-learning evidence
- Telemetry peak memory: simulator `87,575/97,887 MiB` (89.47%); trainer `90,932/97,887 MiB` (92.89%); both reached 100% peak utilization
- Warning: a Ray DataLoader worker shutdown traceback appeared after progress reached 100%; final metrics, manifest, Slurm status, and exit code were all successful

The public evidence tree contains GPU telemetry, allocation/Slurm status, and manifest/submission metadata. Raw trainer, simulator, and role logs remain only on the controlled experiment server because they include generated user profiles and full CAR conversations. Checkpoint tensors and serialized runtime state are also excluded.
