import sys

sys.path.append("../..")

from config.config import ExpArgs
from config.types_enums import RefTokenNameTypes
from ablations.times.main_times import BaselinesTimes
from runs.runs_utils import get_task
from utils.utils_functions import is_model_encoder_only



arg_task = "sst"
arg_explained_model_backbone = "BERT"
arg_attribution_score_function = "alti"
arg_metric = "COMPREHENSIVENESS"
arg_solvex_beam_size = 10

ExpArgs.task = get_task(arg_task)
ExpArgs.explained_model_backbone = arg_explained_model_backbone

ExpArgs.ref_token_name = RefTokenNameTypes.MASK.value if is_model_encoder_only() else RefTokenNameTypes.UNK.value
ExpArgs.BEAM_SIZE = arg_solvex_beam_size
ExpArgs.eval_metric = arg_metric



print("*" * 20, arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
BaselinesTimes(attr_score_function = arg_attribution_score_function, data_limit = 10).run()
print("*" * 20, "END OF ", arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
