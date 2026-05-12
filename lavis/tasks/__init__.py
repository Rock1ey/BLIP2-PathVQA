from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.tasks.vqa import GQATask, VQATask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "GQATask",
    "VQATask",
]
