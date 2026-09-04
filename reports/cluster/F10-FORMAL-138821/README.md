# F10 Formal Continuous Run — Job 138821

Job `138821` restored the complete step-100 training state and reached step 250 on one node with two RTX PRO 6000 GPUs. The training task and final checkpoint succeeded, while the batch allocation returned `FAILED/1:0` because the original post-run pruning utility rejected veRL's incomplete step-150/200 tombstone directories.

Steps 101--250 contained 76/150 effective outcome-gradient steps with mean batch reward `0.117139`. Across all 250 F10 steps, 114 steps carried effective outcome gradients. CAR dev mean@1 was `0.230769`, `0.269231`, `0.230769`, and `0.230769` at steps 100, 150, 200, and 250, respectively, so the run does not support a final performance-improvement claim.

The pruning utility was updated to distinguish complete resumable checkpoints from older incomplete tombstones while preserving strict marker, keep-step, path, symlink, and newer-step safety checks. After local and remote regression tests passed, the user approved deletion of step 100, 150, and 200. The post-remediation audit reports one complete checkpoint: step 250, 11 files, `31,443,788,637` bytes, marker 250.

Raw conversations and model weights remain on the controlled server and are not published. `summary.json` contains the public aggregate metrics.
