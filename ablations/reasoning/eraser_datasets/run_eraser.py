import argparse
import sys

sys.path.append("")
sys.path.append("../../..")

import pandas as pd

from config.types_enums import ModelBackboneTypes, RefTokenNameTypes
from ablations.reasoning.eraser_datasets.main_items.main_item import BaselinesItem
from config.config import ExpArgs
from config.tasks import IMDB_TASK

from utils.utils_functions import is_model_encoder_only

parser = argparse.ArgumentParser(description = 'Argument parser')

parser.add_argument('attribution_score_function', type = str, help = '')

args = parser.parse_args()

arg_task = IMDB_TASK
arg_explained_model_backbone = ModelBackboneTypes.BERT.value
arg_attribution_score_function = args.attribution_score_function

ExpArgs.task = IMDB_TASK
ExpArgs.explained_model_backbone = ModelBackboneTypes.BERT.value
ExpArgs.ref_token_name = RefTokenNameTypes.MASK.value if is_model_encoder_only() else RefTokenNameTypes.UNK.value

print("*" * 20, arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
test = pd.read_json("./data/test.jsonl", lines = True)
BaselinesItem(attr_score_function = arg_attribution_score_function, data = test).run()
print("*" * 20, "END OF ", arg_task, arg_explained_model_backbone, arg_attribution_score_function, "*" * 20)
