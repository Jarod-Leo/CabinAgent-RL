# F10 formal training through step 100

- Training completed 100/100 optimizer steps after an exact step-50 resume.
- Steps 51--100: 17/50 effective outcome-gradient steps and 33/50 zero-advantage steps.
- Cumulative steps 1--100: 38/100 effective outcome-gradient steps.
- Final CAR dev mean@1: 0.230769; this does not establish a performance improvement.
- Mean step time and throughput for steps 51--100: 154.06 seconds and 1394.67 token/s.
- Simulator/trainer peak VRAM: 89.47% / 97.11%; no additional memory utilization increase is safe.
- No NaN, OOM, reward-schema failure, or aborted trajectory was observed.
- The training result is valid, but storage remediation is required: the prior and current full checkpoints both remain because the resumed process did not register the loaded checkpoint in veRL's in-memory retention list.

Raw conversations, internal cluster identifiers, server paths, checkpoint tensors, and serialized runtime state are excluded from this public-safe report.
