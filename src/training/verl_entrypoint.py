"""veRL PPO entrypoint with CabinAgent-RL estimators registered first."""

from __future__ import annotations

import os

from src.training.long_horizon_advantage import register_with_verl


def main() -> None:
    register_with_verl()
    import verl.trainer.main_ppo as main_ppo

    if os.environ.get("CABIN_BEST_CHECKPOINT_ENABLED", "0") == "1":
        from src.training.verl_task_runner import CabinTaskRunnerV1

        main_ppo.TaskRunnerV1 = CabinTaskRunnerV1

    main_ppo.main()


if __name__ == "__main__":
    main()
