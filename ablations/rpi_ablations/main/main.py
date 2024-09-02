import gc
import os
import sys
from enum import Enum
from pathlib import Path
from typing import List

from torch import Tensor

from main.utils.baselines_utils import get_model, get_data, get_tokenizer, init_baseline_exp
from main.utils.baslines_model_functions import ForwardModel, get_inputs
from main.utils.seg_ig import SequentialIntegratedGradients
from utils.dataclasses.evaluations import DataForEvaluation, DataForEvaluationInputs

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from config.config import BackbonesMetaData, ExpArgs
from config.constants import TEXT_PROMPT, LABEL_PROMPT_NEW_LINE
from config.types_enums import RefTokenNameTypes, AttrScoreFunctions, TokenEvaluationOptions, EvalMetric
from utils.utils_functions import (run_model, get_device, is_model_encoder_only, merge_prompts, conv_to_word_embedding,
                                   is_use_prompt)

import torch

from captum.attr import (IntegratedGradients)
from evaluations.evaluations import evaluate_tokens_attributions


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
        ExpArgs.attribution_scores_function = attr_score_function

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

        if AttrScoreFunctions.llm.value == ExpArgs.attribution_scores_function:
            if ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
                model_path = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/INSTRUCT/meta-llama_Llama-2-7b-chat-hf"
            elif ExpArgs.explained_model_backbone == ModelBackboneTypes.MISTRAL.value:
                model_path = f"{LOCAL_MODELS_PREFIX}/DOWNLOADED_MODELS/INSTRUCT/mistralai_Mistral-7B-Instruct-v0.1"
            else:
                raise ValueError("unsupported LLM")
            self.instruct_model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir = HF_CACHE)
            self.instruct_tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir = HF_CACHE)

    def set_prompts(self):
        if is_use_prompt():
            task_prompt = "\n\n".join([self.task.llm_task_prompt, self.task.llm_few_shots_prompt, TEXT_PROMPT])
            self.task_prompt_input_ids, self.task_prompt_attention_mask = self.encode(task_prompt, True)
            self.label_prompt_input_ids, self.label_prompt_attention_mask = self.encode(LABEL_PROMPT_NEW_LINE, False)

            self.task_prompt_input_ids_embeddings = conv_to_word_embedding(self.model, self.task_prompt_input_ids)
            self.label_prompt_input_ids_embeddings = conv_to_word_embedding(self.model, self.label_prompt_input_ids)

            labels_tokens = [self.tokenizer.encode(str(l), return_tensors = "pt", add_special_tokens = False) for l in
                             list(ExpArgs.task.labels_int_str_maps.keys())]

            ExpArgs.label_vocab_tokens = torch.stack(labels_tokens).squeeze()[:, -1]

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

    def run(self):

        if ExpArgs.attribution_scores_function == AttrScoreFunctions.decompX.value:
            sys.path.append(f"{os.getcwd()}/../../main/utils/DecompX")
            from main.utils.decompX_utils import DecomposeXBaseline
        elif ExpArgs.attribution_scores_function == AttrScoreFunctions.alti.value:
            sys.path.append(f"{os.getcwd()}/../../main/utils/transformer-contributions/alti")
            from main.utils.alti_utils import AltiBaseline
        elif (ExpArgs.attribution_scores_function == AttrScoreFunctions.glob_enc.value) or (
                ExpArgs.attribution_scores_function == AttrScoreFunctions.glob_enc_dim_0.value):
            sys.path.append(f"{os.getcwd()}/../../main/utils/GlobEnc")
            from main.utils.globenc_utils import GlobEncBaseline
            self.glob_enc_baseline = GlobEncBaseline
        elif ExpArgs.attribution_scores_function == AttrScoreFunctions.solvability.value:
            # sys.path.append(f"{os.getcwd()}/../../main/utils/solvability_explainer")
            if not is_model_encoder_only():
                self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
                self.tokenizer.padding_side = "left"
            from main.utils.solvability_explainer.solvex.explainer import BeamSearchExplainer
            from main.utils.solvability_explainer.solvex.masker import TextWordMasker

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
            (origin_input_ids, origin_ref_input_ids, origin_input_embed, origin_ref_input_embed, position_embed,
             ref_position_embed, type_embed, ref_type_embed, origin_attention_mask) = get_inputs(
                tokenizer = self.tokenizer, model = self.model, model_name = self.model_name,
                ref_token = self.ref_token, text = txt, device = self.device)
            attention_mask = origin_attention_mask.clone()
            input_ids, attention_mask = self.merge_prompts_handler(origin_input_ids.clone(), attention_mask)
            ref_input_ids, _ = self.merge_prompts_handler(origin_ref_input_ids.clone(), attention_mask)
            input_embed, _ = self.merge_prompts_embeddings__handler(origin_input_embed.clone(), attention_mask)
            ref_input_embed, _ = self.merge_prompts_embeddings__handler(origin_ref_input_embed.clone(), attention_mask)

            with torch.no_grad():
                pred_origin_logits = run_model(model = self.model, input_ids = input_ids,
                                               attention_mask = attention_mask, is_return_logits = True)
                model_pred_origin = torch.argmax(pred_origin_logits, dim = 1)
            self.model.zero_grad()

            attr_scores = None

            if AttrScoreFunctions.deep_lift.value == ExpArgs.attribution_scores_function:
                explainer = DeepLift(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = ref_input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr_scores = summarize_attributions(_attr)

            if AttrScoreFunctions.gradient_shap.value == ExpArgs.attribution_scores_function:
                explainer = GradientShap(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = torch.cat([ref_input_embed, input_embed]),
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr_scores = summarize_attributions(_attr)

            if AttrScoreFunctions.lime.value == ExpArgs.attribution_scores_function:
                explainer = Lime(self.lime_func)
                _attr = explainer.attribute(input_ids, target = pred_origin_logits.max(1)[1])
                attr_scores = _attr.squeeze().detach()

            if AttrScoreFunctions.input_x_gradient.value == ExpArgs.attribution_scores_function:
                explainer = InputXGradient(nn_forward_func)
                _attr = explainer.attribute(input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr_scores = summarize_attributions(_attr)

            if AttrScoreFunctions.integrated_gradients.value == ExpArgs.attribution_scores_function:
                explainer = IntegratedGradients(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = ref_input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr_scores = summarize_attributions(_attr)

            if AttrScoreFunctions.sequential_integrated_gradients.value == ExpArgs.attribution_scores_function:
                explainer = SequentialIntegratedGradients(nn_forward_func)

                n_steps = 50  # default value
                if not is_model_encoder_only():
                    n_steps = 10
                    if ExpArgs.task.name in [IMDB_TASK.name, AGN_TASK.name]:
                        n_steps = 4
                _attr = explainer.attribute(input_embed, baselines = ref_input_embed, n_steps = n_steps,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr_scores = summarize_attributions(_attr).detach()

                del explainer
                del _attr

            if AttrScoreFunctions.decompX.value == ExpArgs.attribution_scores_function:
                decompse = DecomposeXBaseline(self.model_path)
                attr_scores = decompse.compute_attr(input_ids, attention_mask)

            if AttrScoreFunctions.alti.value == ExpArgs.attribution_scores_function:
                alti_input_ids, alti_attention_mask = input_ids.clone(), attention_mask.clone()
                origin_model_max_length = self.tokenizer.model_max_length
                if (not is_model_encoder_only()) and (self.task.name in [IMDB_TASK.name, AGN_TASK.name]):
                    alti_input_ids, alti_attention_mask = self.get_alti_input(txt)

                alti = AltiBaseline(self.model)
                _attr = alti.compute_attr(alti_input_ids, alti_attention_mask)
                attr_scores = summarize_attributions(_attr, sum_dim = 0).detach()

                del alti
                del _attr

                if attr_scores is None:
                    raise ValueError("attr_scores score can not be none")

            gc.collect()
            torch.cuda.empty_cache()

            eval_attr_score = attr_scores
            if is_use_prompt() and (AttrScoreFunctions.llm.value != ExpArgs.attribution_scores_function):
                # print(f"before prompt - eval_attr_score.shape: {eval_attr_score.shape}")
                eval_attr_score = attr_scores[
                                  self.task_prompt_input_ids.shape[-1]:-self.label_prompt_input_ids.shape[-1]].detach()
                # print(f"after prompt - eval_attr_score.shape: {eval_attr_score.shape}")
                # print("\n\n" + "-" * 100)
                # print(self.tokenizer.batch_decode(input_ids)[0])
                # print("\n\n" + "-" * 100)
                print(input_ids)
                print("\n\n" + "^" * 100)
            for metric in self.metrics:
                experiment_path = self.get_folder_name(metric)
                ExpArgs.evaluation_metric = metric.value

                data_for_eval: DataForEvaluation = DataForEvaluation(  #
                    tokens_attributions = eval_attr_score.detach(),  #
                    input = DataForEvaluationInputs(  #
                        input_ids = origin_input_ids,  #
                        attention_mask = origin_attention_mask,  #
                        task_prompt_input_ids = self.task_prompt_input_ids,  #
                        label_prompt_input_ids = self.label_prompt_input_ids,  #
                        task_prompt_attention_mask = self.task_prompt_attention_mask,  #
                        label_prompt_attention_mask = self.label_prompt_attention_mask  #
                    ),  #
                    explained_model_predicted_class = model_pred_origin.squeeze(),  #
                    explained_model_predicted_logits = pred_origin_logits.squeeze())

                metric_result, metric_result_item = evaluate_tokens_attributions(self.model, self.tokenizer, self.ref_token,
                                                                                 data = data_for_eval,
                                                                                 experiment_path = experiment_path,
                                                                                 item_index = f"{i}_{item_id}", )

            gc.collect()
            torch.cuda.empty_cache()

            if ExpArgs.is_save_results:
                with open(Path(experiment_path, "results.csv"), 'a', newline = '', encoding = 'utf-8-sig') as f:
                    metric_result_item.to_csv(f, header = f.tell() == 0, index = False)
