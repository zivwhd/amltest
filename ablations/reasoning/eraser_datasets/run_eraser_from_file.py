import os
import pickle
import sys

from transformers import AutoTokenizer

from config.constants import HF_CACHE
from ablations.reasoning.eraser_datasets.eraser_metrics import main
from ablations.reasoning.eraser_datasets.eraser_utils import scores_per_word_from_scores_per_token

sys.path.append("")
sys.path.append("../../..")

import pandas as pd

from config.types_enums import ModelBackboneTypes, RefTokenNameTypes
from config.config import ExpArgs
from config.tasks import IMDB_TASK

from utils.utils_functions import is_model_encoder_only

arg_task = IMDB_TASK
arg_explained_model_backbone = ModelBackboneTypes.BERT.value

ExpArgs.task = IMDB_TASK
ExpArgs.explained_model_backbone = ModelBackboneTypes.BERT.value
ExpArgs.ref_token_name = RefTokenNameTypes.MASK.value if is_model_encoder_only() else RefTokenNameTypes.UNK.value

print("*" * 20, arg_task, arg_explained_model_backbone, "*" * 20)

tokenizer = AutoTokenizer.from_pretrained(ExpArgs.task.bert_fine_tuned_model, cache_dir = HF_CACHE)

eval_models = [("shap",
                "/home/yonatanto/work/theza/EXPLAINABILITY/EXP_BASELINES/eraser_tests/save_attr_scores/eraser_datasets/OUTPUT/gradient_shap"),
    ("pAML",
     "/home/yonatanto/work/theza/EXPLAINABILITY/EXP_AML/Vtest_eraser/inference/inference_paml/inference_aml/results.pkl"),
    ("sig",
     "/home/yonatanto/work/theza/EXPLAINABILITY/EXP_BASELINES/eraser_tests/save_attr_scores/eraser_datasets/OUTPUT/sequential_integrated_gradients"),
    ("alti",
     "/home/yonatanto/work/theza/EXPLAINABILITY/EXP_BASELINES/eraser_tests/save_attr_scores/eraser_datasets/OUTPUT/alti")]

for model in eval_models:
    df_attr_scores = pd.read_pickle(model[1])
    all_scores = []
    for k in range(5, 85, 5):
        preds, gt = [], []
        loop_over = df_attr_scores.iterrows() if model[0] == "pAML" else enumerate(df_attr_scores)
        for index, row in loop_over:
            if index > 25:
                break
            txt = row["document"]
            attr_scores = row["tokens_attr"] if model[0] == "pAML" else row["attr_scores"]
            input_ids = tokenizer.encode(txt, truncation = True, return_tensors = "pt")
            scores_per_word = scores_per_word_from_scores_per_token(row["document"].split(), tokenizer,
                                                                    input_ids.squeeze(), attr_scores.squeeze())
            hard_rationales = []

            tok_k = int((k / 100) * attr_scores.shape[-1])
            print("calculating top ", tok_k)
            _, indices = scores_per_word.topk(k = tok_k)
            for index in indices.tolist():
                hard_rationales.append({"start_token": index, "end_token": index + 1})
            pred = {"annotation_id": row["annotation_id"],
                    "rationales": [{"docid": row["annotation_id"], "hard_rationale_predictions": hard_rationales}], }
            preds.append(pred)
            gt.append(row)

        scores = main(preds, gt)
        scores["k"] = k
        all_scores.append(scores)
    os.makedirs(f"OUTPUT", exist_ok = True)
    # Open the file in binary write mode
    with open(f"OUTPUT/{model[0]}.pkl", 'wb') as f:
        # Dump the data into the file using pickle
        pickle.dump(all_scores, f)

print("*" * 20, "END OF ", "*" * 20)
