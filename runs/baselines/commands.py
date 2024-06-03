from config.tasks import IMDB_TASK, EMOTION_TASK, SST_TASK, AGN_TASK, RTN_TASK
from config.types_enums import ModelBackboneTypes, AttrScoreFunctions, EvalMetric

tasks = [SST_TASK, RTN_TASK, AGN_TASK, IMDB_TASK]
attr_scores = [AttrScoreFunctions.deep_lift, AttrScoreFunctions.gradient_shap, AttrScoreFunctions.input_x_gradient,
               AttrScoreFunctions.llm]
for attr_score_function in attr_scores:
    for task in tasks:
        for model in [ModelBackboneTypes.LLAMA, ModelBackboneTypes.MISTRAL]:
            print(f"python run_baselines.py {task.name} {attr_score_function.value} {model.name} all")

print("-" * 30)

# tasks = [EMOTION_TASK, SST_TASK, RTN_TASK, AGN_TASK, IMDB_TASK]
# attr_scores = [AttrScoreFunctions.sequential_integrated_gradients, AttrScoreFunctions.alti,
#                AttrScoreFunctions.input_x_gradient, AttrScoreFunctions.gradient_shap, AttrScoreFunctions.deep_lift,
#                AttrScoreFunctions.llm]
# attr_scores = [AttrScoreFunctions.sequential_integrated_gradients]
# for model in [ModelBackboneTypes.LLAMA, ModelBackboneTypes.MISTRAL]:
#     for task in tasks:
#         for attr_score_function in attr_scores:
#             eval_metrics = ["all"]
#             for eval_metric in eval_metrics:
#                 print(f"python run_baselines.py {task.name} {attr_score_function.value} {model.name} {eval_metric}")
