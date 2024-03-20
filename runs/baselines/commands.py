from config.tasks import IMDB_TASK, EMOTION_TASK, SST_TASK, AGN_TASK, RTN_TASK
from config.types_enums import ModelBackboneTypes, AttrScoreFunctions
from utils.utils_functions import is_model_encoder_only

for model in ModelBackboneTypes:
    tasks = [IMDB_TASK, EMOTION_TASK, SST_TASK, AGN_TASK, RTN_TASK]
    if not is_model_encoder_only(model.value):
        tasks = [IMDB_TASK, SST_TASK, AGN_TASK]
    for task in [IMDB_TASK, EMOTION_TASK, SST_TASK, AGN_TASK, RTN_TASK]:
        for attr_score_function in AttrScoreFunctions:
            print(
                f"python run_baselines.py {task.name} {attr_score_function.value} {model.name} all")  # print(f"python run_eda.py {task.name} {model.name}")
