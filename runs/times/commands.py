from config.tasks import IMDB_TASK, EMOTION_TASK, SST_TASK, AGN_TASK, RTN_TASK
from config.types_enums import ModelBackboneTypes, AttrScoreFunctions, EvalMetric

tasks = [EMOTION_TASK, SST_TASK, RTN_TASK, AGN_TASK, IMDB_TASK]
attr_scores = AttrScoreFunctions
for task in tasks:
    for model in [ModelBackboneTypes.BERT, ModelBackboneTypes.ROBERTA, ModelBackboneTypes.DISTILBERT]:
        for attr_score_function in attr_scores:
            eval_metrics = ["all"]

            if attr_score_function.value == AttrScoreFunctions.llm.value:
                continue
            elif attr_score_function.value == AttrScoreFunctions.solvability.value:
                eval_metrics = [item.name for item in EvalMetric]
                if task.name in [AGN_TASK.name, IMDB_TASK.name]:
                    continue
            for eval_metric in eval_metrics:
                if (attr_score_function.value == AttrScoreFunctions.solvability.value):
                    print(f"python run_baselines.py {task.name} {attr_score_function.value} {model.name} {eval_metric}")

print("-" * 30)

tasks = [EMOTION_TASK, SST_TASK, RTN_TASK, AGN_TASK, IMDB_TASK]
attr_scores = [AttrScoreFunctions.input_x_gradient, AttrScoreFunctions.gradient_shap, AttrScoreFunctions.deep_lift,
               AttrScoreFunctions.llm]
for task in tasks:
    for model in [ModelBackboneTypes.LLAMA, ModelBackboneTypes.MISTRAL]:
        for attr_score_function in attr_scores:
            eval_metrics = ["all"]
            for eval_metric in eval_metrics:
                print(f"python run_baselines.py {task.name} {attr_score_function.value} {model.name} {eval_metric}")
