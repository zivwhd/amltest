import sys

from config.config import ExpArgs
from config.tasks import IMDB_TASK

sys.path.append("")
sys.path.append("../../..")
# sys.path.append("../..")
import pandas as pd

from config.types_enums import AttrScoreFunctions, ModelBackboneTypes
from ablations.reasoning.eraser_datasets.main_items.main_item import BaselinesItem

test = pd.read_json("./data/test.jsonl", lines = True)

ExpArgs.task = IMDB_TASK
ExpArgs.explained_model_backbone = ModelBackboneTypes.BERT.value
BaselinesItem(attr_score_function = AttrScoreFunctions.input_x_gradient.value, data = test).run()
