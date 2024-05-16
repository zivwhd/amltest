from config.tasks import IMDB_TASK, EMOTION_TASK, SST_TASK, AGN_TASK, RTN_TASK
from config.types_enums import ModelBackboneTypes, AttrScoreFunctions, EvalMetric

tasks = [EMOTION_TASK, SST_TASK, RTN_TASK, AGN_TASK, IMDB_TASK]
attr_scores = [AttrScoreFunctions.llm]
for model in [ModelBackboneTypes.LLAMA, ModelBackboneTypes.MISTRAL]:
    print("#" * 200)
    for task in tasks:
        for attr_score_function in attr_scores:
            eval_metrics = ["all"]
            for eval_metric in eval_metrics:
                print(f"python run_baselines.py {task.name} {attr_score_function.value} {model.name} {eval_metric}")
