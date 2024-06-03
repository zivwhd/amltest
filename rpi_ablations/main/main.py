import gc
import os
from enum import Enum
from pathlib import Path
from typing import List

from torch import Tensor

from main.utils.baselines_utils import get_model, get_data, get_tokenizer, init_baseline_exp
from main.utils.baslines_model_functions import ForwardModel, get_inputs
from main.utils.seg_ig import SequentialIntegratedGradients
from utils.dataclasses.evaluations import DataForEval, DataForEvalInputs

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from config.config import BackbonesMetaData, ExpArgs
from config.constants import TEXT_PROMPT, LABEL_PROMPT_NEW_LINE
from config.types_enums import RefTokenNameTypes, AttrScoreFunctions, EvalTokens, EvalMetric
from utils.utils_functions import (run_model, get_device, is_model_encoder_only, merge_prompts, conv_to_word_embedding,
                                   is_use_prompt)

import torch

from captum.attr import (IntegratedGradients)
from evaluations.evaluations import evaluate_tokens_attr


def get_alphas_from_timestamp(model, tsteps):
    beta1 = 1e-4
    beta2 = 0.02

    b_t = (beta2 - beta1) * torch.linspace(0, 1, tsteps + 1, device = model.device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim = 0).exp()
    ab_t[0] = 1
    return ab_t


def perturb_input(x, step, noise, ab_t):
    return (ab_t.sqrt()[step, None, None, None] * x.cuda() + (1 - ab_t[step, None, None, None]) * noise).cuda()


def summarize_attributions(attributions, sum_dim = -1):
    attributions = attributions.sum(dim = sum_dim).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


class Baselines:
    def __init__(self, exp_name: str, attr_score_function: str, metrics: List[EvalMetric]):
        self.n_samples = 24
        self.time_steps = 500
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
        ExpArgs.attr_score_function = attr_score_function

        self.device = get_device()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.ref_token = None
        self.set_ref_token()
        self.task_prompt_input_ids = None
        self.label_prompt_input_ids = None
        self.task_prompt_input_ids_embeddings = None
        self.label_prompt_input_ids_embeddings = None
        self.label_prompt_attention_mask = None
        self.task_prompt_attention_mask = None
        self.set_prompts()

    def set_prompts(self):
        if is_use_prompt():
            task_prompt = "\n\n".join([self.task.llm_task_prompt, self.task.llm_few_shots_prompt, TEXT_PROMPT])
            self.task_prompt_input_ids, self.task_prompt_attention_mask = self.encode(task_prompt, True)
            self.label_prompt_input_ids, self.label_prompt_attention_mask = self.encode(LABEL_PROMPT_NEW_LINE, False)

            self.task_prompt_input_ids_embeddings = conv_to_word_embedding(self.model, self.task_prompt_input_ids)
            self.label_prompt_input_ids_embeddings = conv_to_word_embedding(self.model, self.label_prompt_input_ids)

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
        elif ExpArgs.ref_token_name == RefTokenNameTypes.UNK.value:
            self.ref_token = self.tokenizer.unk_token_id
        else:
            raise NotImplementedError

        if not is_model_encoder_only():
            self.tokenizer.pad_token_id = self.ref_token
            self.model.config.pad_token_id = self.ref_token

    def run(self):

        # Prepare forward model
        nn_forward_func = ForwardModel(model = self.model, model_name = self.model_name)
        # Compute attributions

        for metric in self.metrics:
            result_path = self.get_folder_name(metric)
            os.makedirs(result_path, exist_ok = True)

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
            input_embed, _ = self.merge_prompts_embeddings__handler(input_embed, attention_mask)
            ref_input_embed, _ = self.merge_prompts_embeddings__handler(ref_input_embed, attention_mask)

            with torch.no_grad():
                pred_origin_logits = run_model(model = self.model, input_ids = input_ids,
                                               attention_mask = attention_mask, is_return_logits = True)
                model_pred_origin = torch.argmax(pred_origin_logits, dim = 1)

            ab_t = get_alphas_from_timestamp(self.model, self.time_steps)

            for i in range(self.n_samples):
                # start_time = time.time()
                # print(f"i - {i}")
                noise_tensor = torch.normal(mean=ref_input_embed.float())
                # noise_tensor = torch.randn_like(ref_input_embed)
                perturb_baseline = perturb_input(ref_input_embed, self.time_steps, noise_tensor, ab_t)

                self.model.zero_grad()

                attr_scores = None

                if AttrScoreFunctions.integrated_gradients.value == ExpArgs.attr_score_function:
                    explainer = IntegratedGradients(nn_forward_func)
                    _attr = explainer.attribute(input_embed, baselines = perturb_baseline, n_steps = 30,  # for RPI comp
                                                additional_forward_args = (
                                                    attention_mask, position_embed, type_embed,), )
                    attr_scores = summarize_attributions(_attr)

                if AttrScoreFunctions.sequential_integrated_gradients.value == ExpArgs.attr_score_function:
                    explainer = SequentialIntegratedGradients(nn_forward_func)

                    n_steps = 30  # FOR RPI COMP
                    _attr = explainer.attribute(input_embed, baselines = perturb_baseline, n_steps = n_steps,
                                                additional_forward_args = (
                                                    attention_mask, position_embed, type_embed,), )
                    attr_scores = summarize_attributions(_attr).detach()

                    del explainer
                    del _attr

                if attr_scores is None:
                    raise ValueError("attr_scores score can not be none")

                gc.collect()
                torch.cuda.empty_cache()

                eval_attr_score = attr_scores
                if is_use_prompt():
                    eval_attr_score = attr_scores[
                                      self.task_prompt_input_ids.shape[-1]:-self.label_prompt_input_ids.shape[
                                          -1]].detach()

                for metric in self.metrics:
                    if is_model_encoder_only():
                        test_eval_tokens = EvalTokens
                    elif is_use_prompt():
                        test_eval_tokens = [EvalTokens.ALL_TOKENS]
                    else:
                        test_eval_tokens = [EvalTokens.ALL_TOKENS, EvalTokens.NO_SPECIAL_TOKENS]

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

                        metric_result, metric_result_item = evaluate_tokens_attr(self.model, self.tokenizer,
                                                                                 self.ref_token, data = data_for_eval,
                                                                                 experiment_path = experiment_path,
                                                                                 item_index = f"{i}_{item_id}", )

                        gc.collect()
                        torch.cuda.empty_cache()

                        if ExpArgs.is_save_results:
                            with open(Path(experiment_path, "results.csv"), 'a', newline = '',
                                      encoding = 'utf-8-sig') as f:
                                metric_result_item.to_csv(f, header = f.tell() == 0, index = False)

                # end_time = time.time()
                # execution_time = end_time - start_time
                # print("Execution time:", execution_time, "seconds")

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
                             label_prompt_attention_mask = self.label_prompt_attention_mask)

    def merge_prompts_embeddings__handler(self, input_ids: Tensor, attention_mask: Tensor):
        return merge_prompts(inputs = input_ids, attention_mask = attention_mask,
                             task_prompt_input_ids = self.task_prompt_input_ids_embeddings,
                             label_prompt_input_ids = self.label_prompt_input_ids_embeddings,
                             task_prompt_attention_mask = self.task_prompt_attention_mask,
                             label_prompt_attention_mask = self.task_prompt_attention_mask)
