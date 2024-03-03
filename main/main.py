import gc
import os
import sys

from transformers import AutoTokenizer

from main.utils.baselines_utils import get_model, get_data, get_tokenizer
from main.utils.baslines_model_functions import ForwardModel, get_inputs
from main.utils.seg_ig import SequentialIntegratedGradients

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from config.config import BackbonesMetaData, ExpArgs
from config.constants import LABELS_NAME, INPUT_IDS_NAME, ATTENTION_MASK_NAME
from config.types_enums import EvalMetric, RefTokenNameTypes, AttrScoreFunctions
from utils.utils_functions import run_model, get_device
from utils.dataclasses.trainer_outputs import DataForEval

import torch

from captum.attr import (DeepLift, GradientShap, InputXGradient, IntegratedGradients, )
from evaluations.evaluations import evaluate_tokens_attr

sys.path.append(f"{os.getcwd()}/../../main/utils/GlobEnc")
from main.utils.globenc_utils import GlobEncBaseline


def summarize_attributions(attributions, sum_dim = -1):
    attributions = attributions.sum(dim = sum_dim).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


class Baselines:
    def __init__(self, exp_name: str, attr_score_function: str):
        print(f"run {attr_score_function}")
        self.task = ExpArgs.task
        self.exp_path = f"{ExpArgs.default_root_dir}/{exp_name}"
        os.makedirs(self.exp_path, exist_ok = True)
        self.model, self.model_path = get_model()
        self.tokenizer = get_tokenizer(self.model_path)
        self.data = get_data()
        self.model_name = BackbonesMetaData.name[ExpArgs.explained_model_backbone]
        self.attr_score_function = attr_score_function
        self.glob_enc = None

        self.device = get_device()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.ref_token = None
        self.set_ref_token()

    def set_ref_token(self):
        if ExpArgs.ref_token_name == RefTokenNameTypes.MASK.value:
            self.ref_token = self.tokenizer.mask_token_id
        elif ExpArgs.ref_token_name == RefTokenNameTypes.PAD.value:
            self.ref_token = self.tokenizer.pad_token_id
        else:
            raise NotImplementedError

    def run(self):

        if AttrScoreFunctions.decompX.value == self.attr_score_function:
            sys.path.append(f"{os.getcwd()}/../../main/utils/DecompX")
            from main.utils.decompX_utils import DecomposeXBaseline
        elif AttrScoreFunctions.alti.value == self.attr_score_function:
            sys.path.append(f"{os.getcwd()}/../../main/utils/alti")
            from main.utils.alti_utils import AltiBaseline

        # Prepare forward model
        nn_forward_func = ForwardModel(model = self.model, model_name = self.model_name)
        # Compute attributions
        for i, row in enumerate(self.data):
            item_idx = row[2]
            label = row[1]
            txt = row[0]
            (input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed,
             ref_type_embed, attention_mask) = get_inputs(tokenizer = self.tokenizer, model = self.model,
                                                          model_name = self.model_name, ref_token = self.ref_token,
                                                          text = txt, device = self.device)

            with torch.no_grad():
                pred_origin_logits = run_model(model = self.model, model_backbone = ExpArgs.explained_model_backbone,
                                               input_ids = input_ids, attention_mask = attention_mask,
                                               is_return_logits = True)
                model_pred_origin = torch.argmax(pred_origin_logits, dim = 1)
            self.model.zero_grad()

            attr = None

            if AttrScoreFunctions.deep_lift.value == self.attr_score_function:
                explainer = DeepLift(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = ref_input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr = summarize_attributions(_attr)

            if AttrScoreFunctions.gradient_shap.value == self.attr_score_function:
                explainer = GradientShap(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = torch.cat([ref_input_embed, input_embed]),
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr = summarize_attributions(_attr)

            if AttrScoreFunctions.input_x_gradient.value == self.attr_score_function:
                explainer = InputXGradient(nn_forward_func)
                _attr = explainer.attribute(input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr = summarize_attributions(_attr)

            if AttrScoreFunctions.integrated_gradients.value == self.attr_score_function:
                explainer = IntegratedGradients(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = ref_input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr = summarize_attributions(_attr)

            if AttrScoreFunctions.sequential_integrated_gradients.value == self.attr_score_function:
                explainer = SequentialIntegratedGradients(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = ref_input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr = summarize_attributions(_attr)

            if AttrScoreFunctions.decompX.value == self.attr_score_function:
                decompse = DecomposeXBaseline(self.model_path)
                attr = decompse.compute_attr(input_ids, attention_mask)

            if AttrScoreFunctions.alti.value == self.attr_score_function:
                alti = AltiBaseline(self.model)
                _attr = alti.compute_attr(input_ids, attention_mask)
                attr = summarize_attributions(_attr, sum_dim = 0)

            if AttrScoreFunctions.glob_enc.value == self.attr_score_function:
                _attr = self.run_glob_enc(txt, input_ids, attention_mask)
                attr = summarize_attributions(_attr)

            if AttrScoreFunctions.glob_enc_dim_0.value == self.attr_score_function:
                _attr = self.run_glob_enc(txt, input_ids, attention_mask)
                attr = summarize_attributions(_attr, sum_dim = 0)

            if attr is None:
                raise ValueError("attr score can not be none")

            # Clear cuda cache
            default_val = torch.tensor(0)

            outputs: DataForEval = DataForEval(tokens_attr = attr.unsqueeze(0),
                                               input = {INPUT_IDS_NAME: input_ids, ATTENTION_MASK_NAME: attention_mask,
                                                        LABELS_NAME: torch.tensor([label])}, loss = default_val,
                                               pred_loss = default_val, pred_loss_mul = default_val,
                                               tokens_attr_sparse_loss = default_val, pred_origin = model_pred_origin,
                                               pred_origin_logits = pred_origin_logits,
                                               tokens_attr_sparse_loss_mul = default_val,
                                               gt_target = torch.tensor([label]))

            for metric in EvalMetric:
                ExpArgs.eval_metric = metric.value
                evaluate_tokens_attr(self.model, self.tokenizer, self.ref_token, [outputs],
                                     stage = self.attr_score_function, experiment_path = self.exp_path, verbose = True,
                                     item_index = i, is_sequel = True)
                gc.collect()
                torch.cuda.empty_cache()

    def run_glob_enc(self, txt, input_ids, attention_mask):
        if self.glob_enc is None:
            self.glob_enc = GlobEncBaseline(self.model_path, self.model, self.task)
        _attr = self.glob_enc.compute_attr(txt, input_ids, attention_mask)
        _attr = torch.tensor(_attr)
        return _attr
