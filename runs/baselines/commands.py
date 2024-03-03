from config.tasks import IMDB_TASK, EMOTION_TASK, SST_TASK, AGN_TASK, RTN_TASK
from config.types_enums import ModelBackboneTypes, AttrScoreFunctions

for model in ModelBackboneTypes:
    tasks = [IMDB_TASK, EMOTION_TASK, SST_TASK, AGN_TASK, RTN_TASK]
    if model.value == ModelBackboneTypes.LLAMA.value:
        tasks = [IMDB_TASK, SST_TASK, AGN_TASK]
    for task in [IMDB_TASK, EMOTION_TASK, SST_TASK, AGN_TASK, RTN_TASK]:
        for attr_score_function in AttrScoreFunctions:
            print(
                f"python run_baselines.py {task.name} {attr_score_function.value} {model.name}")  # print(f"python run_eda.py {task.name} {model.name}")
