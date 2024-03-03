from config.tasks import IMDB_TASK, EMOTION_TASK, SST_TASK, AGN_TASK, RTN_TASK
from config.types_enums import ModelBackboneTypes

for task in [IMDB_TASK, EMOTION_TASK, SST_TASK, AGN_TASK, RTN_TASK]:
    for model in ModelBackboneTypes:
        print(f"python run_eda.py {task.name} {model.name}")
