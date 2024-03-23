import gc
import os
import sys
from pathlib import Path
from typing import List

from torch import Tensor
from enum import Enum

from main.utils.baselines_utils import get_model, get_data, get_tokenizer, init_baseline_exp
from main.utils.baslines_model_functions import ForwardModel, get_inputs
from main.utils.seg_ig import SequentialIntegratedGradients
from utils.dataclasses.evaluations import DataForEval, DataForEvalInputs

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from config.config import BackbonesMetaData, ExpArgs
from config.constants import TEXT_PROMPT, LABEL_PROMPT
from config.types_enums import RefTokenNameTypes, AttrScoreFunctions, EvalTokens, EvalMetric
from utils.utils_functions import run_model, get_device, is_model_encoder_only, merge_prompts

import torch

from captum.attr import (DeepLift, GradientShap, InputXGradient, IntegratedGradients, )
from evaluations.evaluations import evaluate_tokens_attr


def summarize_attributions(attributions, sum_dim = -1):
    attributions = attributions.sum(dim = sum_dim).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


class Baselines:
    def __init__(self, exp_name: str, attr_score_function: str, metrics: List[EvalMetric]):
        print(f"run {attr_score_function}")
        init_baseline_exp()
        self.task = ExpArgs.task
        self.metrics = metrics
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
        self.task_prompt_input_ids = None
        self.label_prompt_input_ids = None
        self.label_prompt_attention_mask = None
        self.task_prompt_attention_mask = None
        self.set_prompts()

    def set_prompts(self):
        if not is_model_encoder_only():
            task_prompt = "\n\n".join([self.task.llm_task_prompt, self.task.llm_few_shots_prompt, TEXT_PROMPT])
            self.task_prompt_input_ids, self.task_prompt_attention_mask = self.encode(task_prompt, True)
            self.label_prompt_input_ids, self.label_prompt_attention_mask = self.encode(LABEL_PROMPT, False)
            labels_tokens = [self.tokenizer.encode(str(l), return_tensors = "pt", add_special_tokens = False) for l in
                             list(ExpArgs.task.labels_int_str_maps.keys())]

            ExpArgs.labels_tokens_opt = torch.stack(labels_tokens).squeeze()[:, -1]

    def get_folder_name(self, metric: Enum):
        return f"{self.exp_path}/metric_{metric.value}"

    def set_ref_token(self):
        if ExpArgs.ref_token_name == RefTokenNameTypes.MASK.value:
            self.ref_token = self.tokenizer.mask_token_id
        elif ExpArgs.ref_token_name == RefTokenNameTypes.PAD.value:
            self.ref_token = self.tokenizer.pad_token_id
        else:
            raise NotImplementedError

    def run(self):

        if self.attr_score_function == AttrScoreFunctions.decompX.value:
            sys.path.append(f"{os.getcwd()}/../../main/utils/DecompX")
            from main.utils.decompX_utils import DecomposeXBaseline
        elif self.attr_score_function == AttrScoreFunctions.alti.value:
            sys.path.append(f"{os.getcwd()}/../../main/utils/transformer-contributions/alti")
            from main.utils.alti_utils import AltiBaseline
        elif (self.attr_score_function == AttrScoreFunctions.glob_enc.value) or (
                self.attr_score_function == AttrScoreFunctions.glob_enc_dim_0.value):
            sys.path.append(f"{os.getcwd()}/../../main/utils/GlobEnc")
            from main.utils.globenc_utils import GlobEncBaseline
            self.glob_enc_baseline = GlobEncBaseline

        # Prepare forward model
        nn_forward_func = ForwardModel(model = self.model, model_name = self.model_name)
        # Compute attributions

        for metric in self.metrics:
            result_path = self.get_folder_name(metric)
            os.makedirs(result_path, exist_ok = True)

        self.data = self.data[:5]  # TODO - remove
        for i, row in enumerate(self.data):
            item_id = row[2]
            label = row[1]
            txt = row[0]
            (input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed,
             ref_type_embed, attention_mask) = get_inputs(tokenizer = self.tokenizer, model = self.model,
                                                          model_name = self.model_name, ref_token = self.ref_token,
                                                          text = txt, device = self.device)
            input_ids, attention_mask = self.merge_prompts_handler(input_ids, attention_mask)
            ref_input_ids, _ = self.merge_prompts_handler(ref_input_ids, attention_mask)
            input_embed, _ = self.merge_prompts_handler(input_embed, attention_mask)
            ref_input_embed, _ = self.merge_prompts_handler(ref_input_embed, attention_mask)

            with torch.no_grad():
                pred_origin_logits = run_model(model = self.model, input_ids = input_ids,
                                               attention_mask = attention_mask, is_return_logits = True)
                model_pred_origin = torch.argmax(pred_origin_logits, dim = 1)
            self.model.zero_grad()

            attr_scores = None

            if AttrScoreFunctions.deep_lift.value == self.attr_score_function:
                explainer = DeepLift(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = ref_input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr_scores = summarize_attributions(_attr)

            if AttrScoreFunctions.gradient_shap.value == self.attr_score_function:
                explainer = GradientShap(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = torch.cat([ref_input_embed, input_embed]),
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr_scores = summarize_attributions(_attr)

            if AttrScoreFunctions.input_x_gradient.value == self.attr_score_function:
                explainer = InputXGradient(nn_forward_func)
                _attr = explainer.attribute(input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr_scores = summarize_attributions(_attr)

            if AttrScoreFunctions.integrated_gradients.value == self.attr_score_function:
                explainer = IntegratedGradients(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = ref_input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr_scores = summarize_attributions(_attr)

            if AttrScoreFunctions.sequential_integrated_gradients.value == self.attr_score_function:
                explainer = SequentialIntegratedGradients(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = ref_input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr_scores = summarize_attributions(_attr)

            if AttrScoreFunctions.decompX.value == self.attr_score_function:
                decompse = DecomposeXBaseline(self.model_path)
                attr_scores = decompse.compute_attr(input_ids, attention_mask)

            if AttrScoreFunctions.alti.value == self.attr_score_function:
                alti = AltiBaseline(self.model)
                _attr = alti.compute_attr(input_ids, attention_mask)
                attr_scores = summarize_attributions(_attr, sum_dim = 0)

            if AttrScoreFunctions.glob_enc.value == self.attr_score_function:
                _attr = self.run_glob_enc(txt, input_ids, attention_mask)
                attr_scores = summarize_attributions(_attr)

            if AttrScoreFunctions.glob_enc_dim_0.value == self.attr_score_function:
                _attr = self.run_glob_enc(txt, input_ids, attention_mask)
                attr_scores = summarize_attributions(_attr, sum_dim = 0)

            if attr_scores is None:
                raise ValueError("attr_scores score can not be none")

            eval_attr_score = attr_scores
            if not is_model_encoder_only():
                eval_attr_score = attr_scores[
                                  self.task_prompt_input_ids.shape[-1]:-self.label_prompt_input_ids.shape[-1]].detach()

            for metric in EvalMetric:
                test_eval_tokens = EvalTokens if is_model_encoder_only() else [EvalTokens.ALL_TOKENS]
                for eval_token in test_eval_tokens:
                    experiment_path = self.get_folder_name(metric)
                    ExpArgs.eval_metric = metric.value
                    ExpArgs.eval_tokens = eval_token.value

                    data_for_eval: DataForEval = DataForEval(  #
                        tokens_attr = eval_attr_score.detach(),  #
                        input = DataForEvalInputs(  #
                            input_ids = input_ids,  #
                            attention_mask = attention_mask,  #
                            task_prompt_input_ids = self.task_prompt_input_ids,  #
                            label_prompt_input_ids = self.label_prompt_input_ids,  #
                            task_prompt_attention_mask = self.task_prompt_attention_mask,  #
                            label_prompt_attention_mask = self.label_prompt_attention_mask  #
                        ),  #
                        pred_origin = model_pred_origin.squeeze(),  #
                        pred_origin_logits = pred_origin_logits.squeeze(),  #
                        gt_target = torch.tensor(label))

                    metric_result, metric_result_item = evaluate_tokens_attr(self.model, self.tokenizer, self.ref_token,
                                                                             data = data_for_eval,
                                                                             experiment_path = experiment_path,
                                                                             item_index = f"{i}_{item_id}", )

                    gc.collect()
                    torch.cuda.empty_cache()

                    if ExpArgs.is_save_results:
                        with open(Path(experiment_path, "results.csv"), 'a', newline = '', encoding = 'utf-8-sig') as f:
                            metric_result_item.to_csv(f, header = f.tell() == 0, index = False)

    def encode(self, new_txt, is_add_special_tokens):
        tokenized = self.tokenizer.encode_plus(new_txt, truncation = True, add_special_tokens = is_add_special_tokens,
                                               return_tensors = "pt")
        input_ids = tokenized.input_ids.cuda()
        attention_mask = tokenized.attention_mask.cuda()
        return input_ids, attention_mask

    def merge_prompts_handler(self, input_ids: Tensor, attention_mask: Tensor):
        return merge_prompts(inputs = input_ids, attention_mask = attention_mask,
                             task_prompt_input_ids = self.task_prompt_input_ids,
                             label_prompt_input_ids = self.label_prompt_input_ids,
                             task_prompt_attention_mask = self.task_prompt_attention_mask,
                             label_prompt_attention_mask = self.task_prompt_attention_mask)

    def run_glob_enc(self, txt, input_ids, attention_mask):
        if self.glob_enc is None:
            self.glob_enc = self.glob_enc_baseline(self.model_path, self.model, self.task)
        _attr = self.glob_enc.compute_attr(txt, input_ids, attention_mask)
        _attr = torch.tensor(_attr)
        return _attr
