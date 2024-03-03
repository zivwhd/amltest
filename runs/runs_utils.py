from config.config import ExpArgsDefault, ExpArgs
from config.tasks import IMDB_TASK, EMOTION_TASK, SST_TASK, AGN_TASK, RTN_TASK


def reset_conf():
    for k, v in vars(ExpArgsDefault).items():
        if "__" not in k:
            setattr(ExpArgs, k, v)


def get_task(task_name):
    if task_name == IMDB_TASK.name:
        return IMDB_TASK
    elif task_name == EMOTION_TASK.name:
        return EMOTION_TASK
    elif task_name == SST_TASK.name:
        return SST_TASK
    elif task_name == AGN_TASK.name:
        return AGN_TASK
    elif task_name == RTN_TASK.name:
        return RTN_TASK
    raise ValueError(f"{task_name} is a unsupported task")
