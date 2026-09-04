"""Project task runner that installs the best-checkpoint policy inside Ray."""

from __future__ import annotations

from pprint import pprint

import ray
from omegaconf import OmegaConf


@ray.remote
class CabinTaskRunnerV1:
    """Small extension of veRL's V1 task runner for project-local hooks."""

    def __init__(self) -> None:
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None
        self.best_checkpoint_controller = None

    def init_agent_loop_manager(self) -> None:
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ
        from verl.utils.import_utils import load_class_from_fqn

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get(
            "agent_loop_manager_class"
        )
        manager_class = (
            load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
            if manager_class_fqn
            else AgentLoopManagerTQ
        )
        self.agent_loop_manager = manager_class.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config) -> None:
        from verl.trainer.ppo.v1 import get_trainer_cls
        from verl.utils.logging_utils import configure_verl_logging

        from src.training.best_checkpoint import install_best_checkpoint_controller
        from src.training.long_horizon_advantage import register_with_verl

        configure_verl_logging()
        register_with_verl()
        import transfer_queue as tq

        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config

        tq.init(config.transfer_queue)
        succeeded = False
        try:
            trainer_cls = get_trainer_cls(config.trainer.v1.trainer_mode)
            self.trainer = trainer_cls(config=config)
            self.trainer.init()
            if bool(config.trainer.get("cabin_best_checkpoint", {}).get("enabled", False)):
                self.best_checkpoint_controller = install_best_checkpoint_controller(self.trainer)
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()
