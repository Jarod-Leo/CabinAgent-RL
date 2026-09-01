"""veRL PPO entrypoint with CabinAgent-RL estimators registered first."""

from __future__ import annotations

from src.training.long_horizon_advantage import register_with_verl


def main() -> None:
    register_with_verl()
    from verl.trainer.main_ppo import main as verl_main

    verl_main()


if __name__ == "__main__":
    main()
