import sys

from main.print_scores import ScoresBaselines

sys.path.append("../..")

from config.config import ExpArgs
from config.types_enums import RefTokenNameTypes, EvalMetric, ModelBackboneTypes, AttrScoreFunctions
from main.main import Baselines
from runs.runs_utils import get_task
from utils.utils_functions import get_current_time, is_model_encoder_only

for arg_task in ["sst", "emotions", "rtn"]:
    arg_explained_model_backbone = ModelBackboneTypes.ROBERTA.value
    arg_attribution_score_function = [  #
        AttrScoreFunctions.sequential_integrated_gradients.value,  #
        AttrScoreFunctions.alti.value  #
    ]

    ExpArgs.task = get_task(arg_task)
    ExpArgs.explained_model_backbone = arg_explained_model_backbone

    ExpArgs.task.test_sample = 150

    ExpArgs.ref_token_name = RefTokenNameTypes.MASK.value if is_model_encoder_only() else RefTokenNameTypes.UNK.value

    exp_path = f"{ExpArgs.task.name}_{ExpArgs.explained_model_backbone}_{arg_attribution_score_function}_{get_current_time()}"
    print("*" * 20, arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
    ScoresBaselines(exp_name = exp_path, attr_score_functions = arg_attribution_score_function).run()
    print("*" * 20, "END OF ", arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
