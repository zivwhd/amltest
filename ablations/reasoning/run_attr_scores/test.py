import sys

sys.path.append("../../..")

from ablations.reasoning.eraser_datasets.eraser_utils import load_jsonl
from ablations.reasoning.run_attr_scores.main_reasoning import BaselinesReasoning

from config.config import ExpArgs
from config.types_enums import RefTokenNameTypes
from runs.runs_utils import get_task
from utils.utils_functions import is_model_encoder_only


arg_task = "sst"
arg_explained_model_backbone = "BERT"
arg_attribution_score_function = "alti"
arg_metric ="COMPREHENSIVENESS"
arg_solvex_beam_size = 10
arg_set_type ="val"

ExpArgs.task = get_task(arg_task)
ExpArgs.explained_model_backbone = arg_explained_model_backbone

ExpArgs.ref_token_name = RefTokenNameTypes.MASK.value if is_model_encoder_only() else RefTokenNameTypes.UNK.value
ExpArgs.BEAM_SIZE = int(arg_solvex_beam_size)
ExpArgs.eval_metric = arg_metric

data = load_jsonl(
    f"/home/yonatanto/work/theza/EMNLP_EXPLAINABILITY/earaser_data/postprocess_movies_data/{arg_set_type}.jsonl")
data = [item["document"] for item in data]

print("*" * 20, arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
BaselinesReasoning(attr_score_function = arg_attribution_score_function).run(data)
print("*" * 20, "END OF ", arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
