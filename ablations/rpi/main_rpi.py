import gc
import os
import time
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from main.utils.baselines_utils import get_model, get_data, get_tokenizer, init_baseline_exp
from main.utils.baslines_model_functions import ForwardModel, get_inputs
from main.utils.seg_ig import SequentialIntegratedGradients
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.dataclasses.evaluations import DataForEvaluation, DataForEvaluationInputs

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from config.config import BackbonesMetaData, ExpArgs
from config.constants import (TEXT_PROMPT, LABEL_PROMPT_NEW_LINE, LOCAL_MODELS_PREFIX, HF_CACHE)
from config.types_enums import RefTokenNameTypes, AttrScoreFunctions, EvalMetric, ModelBackboneTypes
from utils.utils_functions import (run_model, get_device, is_model_encoder_only, merge_prompts, conv_to_word_embedding,
                                   is_use_prompt)

import torch

from captum.attr import (IntegratedGradients)
from evaluations.evaluations import evaluate_tokens_attributions


def summarize_attributions(attributions, sum_dim = -1):
    attributions = attributions.sum(dim = sum_dim).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


class BaselinesRpi:
    def __init__(self, exp_name: str, attr_score_function: str, metrics: List[EvalMetric]):
        self.NUM_SAMPLES = 24
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

        self.glob_enc = None

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

            ExpArgs.label_vocab_tokens = torch.stack(labels_tokens).squeeze()
            if ExpArgs.label_vocab_tokens.ndim != 1:
                raise ValueError("label_vocab_tokens must work with one token only")
            print(f"ExpArgs.label_vocab_tokens: {ExpArgs.label_vocab_tokens}")

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
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def run(self):

        # Prepare forward model
        nn_forward_func = ForwardModel(model = self.model, model_name = self.model_name,

                                       task_prompt_input_ids = self.task_prompt_input_ids,
                                       label_prompt_input_ids = self.label_prompt_input_ids,
                                       task_prompt_attention_mask = self.task_prompt_attention_mask,
                                       label_prompt_attention_mask = self.label_prompt_attention_mask

                                       )
        # Compute attributions

        for metric in self.metrics:
            result_path = self.get_folder_name(metric)
            os.makedirs(result_path, exist_ok = True)

        times = []
        for i, row in enumerate(self.data):
            item_id = row[2]
            txt = row[0]
            (origin_input_ids, origin_ref_input_ids, origin_input_embed, origin_ref_input_embed, position_embed,
             ref_position_embed, type_embed, ref_type_embed, origin_attention_mask) = get_inputs(
                tokenizer = self.tokenizer, model = self.model, model_name = self.model_name,
                ref_token = self.ref_token, text = txt, device = self.device)
            attention_mask = origin_attention_mask.clone()
            merged_input_ids, merged_attention_mask = self.merge_prompts_handler(origin_input_ids.clone(),
                                                                                 attention_mask)
            input_embed = origin_input_embed
            ref_input_embed = origin_ref_input_embed

            print(f"merged_input_ids: {merged_input_ids}")
            with torch.no_grad():
                explained_model_logits = run_model(model = self.model, input_ids = merged_input_ids,
                                                   attention_mask = merged_attention_mask, is_return_logits = True)
                explained_model_predicted_class = torch.argmax(explained_model_logits, dim = 1)

            for sample_idx in range(self.NUM_SAMPLES):
                    ref_input_embed = torch.randn_like(ref_input_embed)

                    self.model.zero_grad()

                    begin = time.time()

                    attribution_scores = None

                    if ExpArgs.attribution_scores_function == AttrScoreFunctions.integrated_gradients.value:
                        explainer = IntegratedGradients(nn_forward_func)
                        _attr = explainer.attribute(input_embed, baselines = ref_input_embed,
                                                    additional_forward_args = (attention_mask, position_embed, type_embed,), )
                        attribution_scores = summarize_attributions(_attr)

                    if ExpArgs.attribution_scores_function == AttrScoreFunctions.sequential_integrated_gradients.value:
                        explainer = SequentialIntegratedGradients(nn_forward_func)

                        n_steps = 50  # default value

                        if is_model_encoder_only():
                            print("C"*100)
                            _attr = explainer.attribute(input_embed,
                                                        internal_batch_size = 3,
                                                        baselines = ref_input_embed, n_steps = n_steps,
                                                        additional_forward_args = (
                                                            attention_mask, position_embed, type_embed,), )
                        elif not is_use_prompt():
                            print("B"*100)
                            n_steps = ExpArgs.n_steps
                            batch_size = int(ExpArgs.batch_size)
                            if batch_size != 0:
                                print("K" * 100)
                                _attr = explainer.attribute(input_embed,
                                                            internal_batch_size = batch_size,
                                                            baselines = ref_input_embed, n_steps = n_steps,
                                                            additional_forward_args = (
                                                            attention_mask, position_embed, type_embed,))
                            else:
                                print("L" * 100)
                                _attr = explainer.attribute(input_embed,
                                                            baselines = ref_input_embed, n_steps = n_steps,
                                                            additional_forward_args = (
                                                            attention_mask, position_embed, type_embed,))

                        else:
                            print("A"*100)
                            n_steps = ExpArgs.n_steps
                            batch_size = int(ExpArgs.batch_size)
                            if batch_size != 0:
                                print("T"*100)
                                _attr = explainer.attribute(input_embed,
                                                            internal_batch_size = batch_size,
                                                            baselines = ref_input_embed, n_steps = n_steps,
                                                            additional_forward_args = (
                                                                attention_mask, position_embed, type_embed,), )
                            else:
                                print("Z"*100)
                                _attr = explainer.attribute(input_embed,
                                                            baselines = ref_input_embed, n_steps = n_steps,
                                                            additional_forward_args = (
                                                                attention_mask, position_embed, type_embed,), )

                        attribution_scores = summarize_attributions(_attr).detach()

                        del explainer
                        del _attr

                    if attribution_scores is None:
                        raise ValueError("attribution_scores score can not be none")

                    gc.collect()
                    torch.cuda.empty_cache()

                    eval_attr_score = attribution_scores

                    if ExpArgs.is_evaluate:
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
                                explained_model_predicted_class = explained_model_predicted_class.squeeze(),  #
                                explained_model_predicted_logits = explained_model_logits.squeeze())

                            evaluation_result, evaluation_item = evaluate_tokens_attributions(self.model, self.tokenizer,
                                                                                              self.ref_token,
                                                                                              data = data_for_eval,
                                                                                              experiment_path = experiment_path,
                                                                                              item_index = f"{i}_{item_id}")

                            gc.collect()
                            torch.cuda.empty_cache()

                            if ExpArgs.is_save_results:
                                evaluation_item["__input_text__"] = txt
                                with open(Path(experiment_path, "results.csv"), 'a', newline = '', encoding = 'utf-8-sig') as f:
                                    evaluation_item.to_csv(f, header = f.tell() == 0, index = False)

                    end = time.time()
                    times.append(end - begin)

                    print(f"duration: {np.array(times).mean()}")

                    if ExpArgs.is_save_times:
                        pd.DataFrame(dict(  #
                            time = [np.array(times).mean()],  #
                            times = [times],  #
                            task = [ExpArgs.task.name],  #
                            model = [ExpArgs.explained_model_backbone],  #
                            amount = [ExpArgs.task.test_sample]  #
                        )).to_csv(f"{self.exp_path}/times_{time.time()}.csv")

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
