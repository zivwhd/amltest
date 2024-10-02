import argparse
import sys

sys.path.append("../..")

from config.config import ExpArgs
from config.types_enums import RefTokenNameTypes, DefaultEvalMetric
from main.main_2 import Baselines
from runs.runs_utils import get_task
from utils.utils_functions import get_current_time, is_model_encoder_only

from ablations.reasoning.eraser_utils import load_jsonl

parser = argparse.ArgumentParser(description = 'Argument parser')

arg_task = "sst"
arg_explained_model_backbone = "BERT"
arg_attribution_score_function = "solvability"
# arg_metric = "COMPREHENSIVENESS"
arg_metric = DefaultEvalMetric

ExpArgs.task = get_task(arg_task)
ExpArgs.explained_model_backbone = arg_explained_model_backbone
ExpArgs.ref_token_name = RefTokenNameTypes.MASK.value if is_model_encoder_only() else RefTokenNameTypes.UNK.value
ExpArgs.SOLVABILITY_BATCH_SIZE = 50
start_data = int(1)
end_data = int(3)


data = load_jsonl(f"/RG/rg-barkan/yonatanto/theza/EMNLP_EXPLAINABILITY/earaser_data/postprocess_movies_data/test.jsonl")
data = [(item["document"], None, idx) for idx, item in enumerate(data)]
data = data[start_data:end_data]

exp_path = f"{ExpArgs.task.name}_{ExpArgs.explained_model_backbone}_{arg_attribution_score_function}_{get_current_time()}"
print("*" * 20, arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
Baselines(exp_name = exp_path, attr_score_function = arg_attribution_score_function, metrics = arg_metric,
          data = data).run()
print("*" * 20, "END OF ", arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
