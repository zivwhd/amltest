import sys

sys.path.append("../..")

from config.config import ExpArgs
from config.types_enums import RefTokenNameTypes, ModelBackboneTypes, AttrScoreFunctions, DefaultEvalMetric
from ablations.rpi.main_rpi import BaselinesRpi
from runs.runs_utils import get_task
from utils.utils_functions import get_current_time, is_model_encoder_only

arg_task = "rtn"
arg_explained_model_backbone = ModelBackboneTypes.BERT.value
arg_attribution_score_function = AttrScoreFunctions.input_x_gradient.value
arg_metric = DefaultEvalMetric

ExpArgs.task = get_task(arg_task)
ExpArgs.explained_model_backbone = arg_explained_model_backbone

ExpArgs.task.test_sample = 30
# ExpArgs.task.is_llm_set_max_len = True
# ExpArgs.task.llm_explained_tokenizer_max_length = 15

ExpArgs.ref_token_name = RefTokenNameTypes.MASK.value if is_model_encoder_only() else RefTokenNameTypes.UNK.value
ExpArgs.BEAM_SIZE = 50
exp_path = f"{ExpArgs.task.name}_{ExpArgs.explained_model_backbone}_{arg_attribution_score_function}_{get_current_time()}"
print("*" * 20, arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
BaselinesRpi(exp_name = exp_path, attr_score_function = arg_attribution_score_function, metrics = arg_metric).run()
print("*" * 20, "END OF ", arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
