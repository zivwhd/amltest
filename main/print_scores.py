import gc
import os
import sys
import time
from enum import Enum
from typing import List

import pandas as pd
from torch import Tensor

from config.tasks import IMDB_TASK, AGN_TASK
from main.utils.baselines_utils import get_model, get_data, get_tokenizer, init_baseline_exp
from main.utils.baslines_model_functions import ForwardModel, get_inputs
from main.utils.seg_ig import SequentialIntegratedGradients

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from config.config import BackbonesMetaData, ExpArgs
from config.constants import (TEXT_PROMPT, LABEL_PROMPT_NEW_LINE)
from config.types_enums import RefTokenNameTypes, AttrScoreFunctions, EvalMetric, ModelBackboneTypes
from utils.utils_functions import (run_model, get_device, is_model_encoder_only, merge_prompts, conv_to_word_embedding,
                                   is_use_prompt)

import torch

from captum.attr import (DeepLift, GradientShap, InputXGradient, IntegratedGradients, Lime)


def summarize_attributions(attributions, sum_dim = -1):
    attributions = attributions.sum(dim = sum_dim).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


class ScoresBaselines:
    def __init__(self, exp_name: str, attr_score_functions: List[str]):
        print(f"run {attr_score_functions}")
        init_baseline_exp()
        self.task = ExpArgs.task
        self.exp_path = f"{ExpArgs.default_root_dir}/{exp_name}"
        os.makedirs(self.exp_path, exist_ok = True)
        self.model, self.model_path = get_model()
        self.tokenizer = get_tokenizer(self.model_path)
        self.data = get_data()
        self.model_name = BackbonesMetaData.name[ExpArgs.explained_model_backbone]
        self.attr_score_functions = attr_score_functions
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

        if AttrScoreFunctions.decompX.value in self.attr_score_functions:
            sys.path.append(f"{os.getcwd()}/../../main/utils/DecompX")
            from main.utils.decompX_utils import DecomposeXBaseline
        elif AttrScoreFunctions.alti.value in self.attr_score_functions:
            sys.path.append(f"{os.getcwd()}/../../main/utils/transformer-contributions/alti")
            from main.utils.alti_utils import AltiBaseline
        elif (AttrScoreFunctions.glob_enc.value in self.attr_score_functions) or (
                AttrScoreFunctions.glob_enc_dim_0.value in self.attr_score_functions):
            sys.path.append(f"{os.getcwd()}/../../main/utils/GlobEnc")
            from main.utils.globenc_utils import GlobEncBaseline
            self.glob_enc_baseline = GlobEncBaseline
        elif AttrScoreFunctions.solvability.value in self.attr_score_functions:
            # sys.path.append(f"{os.getcwd()}/../../main/utils/solvability_explainer")
            if not is_model_encoder_only():
                self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
                self.tokenizer.padding_side = "left"
            from main.utils.solvability_explainer.solvex.explainer import BeamSearchExplainer
            from main.utils.solvability_explainer.solvex.masker import TextWordMasker

        # Prepare forward model
        nn_forward_func = ForwardModel(model = self.model, model_name = self.model_name)
        # Compute attributions

        df_data = []
        time_str = time.time()
        for i, row in enumerate(self.data):
            print("-" * 100)
            if len(row[0].split()) < 6:
                continue
            print(f"{row[0]}")
            df_data_item = {a: [] for a in self.attr_score_functions}
            df_data_item["txt"] = row[0]
            df_data.append(df_data_item)
            pd.DataFrame(df_data).to_pickle(f"./{self.task.name}_{time_str}.pkl")

            for attr_score_function in self.attr_score_functions:
                ExpArgs.attr_score_function = attr_score_function
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
                ref_input_embed, _ = self.merge_prompts_embeddings__handler(origin_ref_input_embed.clone(),
                                                                            attention_mask)

                with torch.no_grad():
                    pred_origin_logits = run_model(model = self.model, input_ids = input_ids,
                                                   attention_mask = attention_mask, is_return_logits = True)
                    model_pred_origin = torch.argmax(pred_origin_logits, dim = 1)
                self.model.zero_grad()

                attr_scores = None

                if AttrScoreFunctions.deep_lift.value == ExpArgs.attr_score_function:
                    explainer = DeepLift(nn_forward_func)
                    _attr = explainer.attribute(input_embed, baselines = ref_input_embed, additional_forward_args = (
                        attention_mask, position_embed, type_embed,), )
                    attr_scores = summarize_attributions(_attr)

                if AttrScoreFunctions.gradient_shap.value == ExpArgs.attr_score_function:
                    explainer = GradientShap(nn_forward_func)
                    _attr = explainer.attribute(input_embed, baselines = torch.cat([ref_input_embed, input_embed]),
                                                additional_forward_args = (
                                                    attention_mask, position_embed, type_embed,), )
                    attr_scores = summarize_attributions(_attr)

                if AttrScoreFunctions.lime.value == ExpArgs.attr_score_function:
                    explainer = Lime(self.lime_func)
                    _attr = explainer.attribute(input_ids, target = pred_origin_logits.max(1)[1])
                    attr_scores = _attr.squeeze().detach()

                if AttrScoreFunctions.input_x_gradient.value == ExpArgs.attr_score_function:
                    explainer = InputXGradient(nn_forward_func)
                    _attr = explainer.attribute(input_embed, additional_forward_args = (
                        attention_mask, position_embed, type_embed,), )
                    attr_scores = summarize_attributions(_attr)

                if AttrScoreFunctions.integrated_gradients.value == ExpArgs.attr_score_function:
                    explainer = IntegratedGradients(nn_forward_func)
                    _attr = explainer.attribute(input_embed, baselines = ref_input_embed, additional_forward_args = (
                        attention_mask, position_embed, type_embed,), )
                    attr_scores = summarize_attributions(_attr)

                if AttrScoreFunctions.sequential_integrated_gradients.value == ExpArgs.attr_score_function:
                    explainer = SequentialIntegratedGradients(nn_forward_func)

                    n_steps = 50  # default value
                    if not is_model_encoder_only():
                        n_steps = 10
                        if ExpArgs.task.name in [IMDB_TASK.name, AGN_TASK.name]:
                            n_steps = 4
                    _attr = explainer.attribute(input_embed, baselines = ref_input_embed, n_steps = n_steps,
                                                additional_forward_args = (
                                                    attention_mask, position_embed, type_embed,), )
                    attr_scores = summarize_attributions(_attr).detach()

                    del explainer
                    del _attr

                if AttrScoreFunctions.decompX.value == ExpArgs.attr_score_function:
                    decompse = DecomposeXBaseline(self.model_path)
                    attr_scores = decompse.compute_attr(input_ids, attention_mask)

                if AttrScoreFunctions.alti.value == ExpArgs.attr_score_function:
                    alti_input_ids, alti_attention_mask = input_ids.clone(), attention_mask.clone()
                    origin_model_max_length = self.tokenizer.model_max_length
                    if (not is_model_encoder_only()) and (self.task.name in [IMDB_TASK.name, AGN_TASK.name]):
                        alti_input_ids, alti_attention_mask = self.get_alti_input(txt)

                    alti = AltiBaseline(self.model)
                    _attr = alti.compute_attr(alti_input_ids, alti_attention_mask)
                    attr_scores = summarize_attributions(_attr, sum_dim = 0).detach()

                    del alti
                    del _attr

                    if origin_model_max_length != self.tokenizer.model_max_length:
                        self.tokenizer.model_max_length = origin_model_max_length

                if AttrScoreFunctions.glob_enc.value == ExpArgs.attr_score_function:
                    _attr = self.run_glob_enc(txt, input_ids, attention_mask)
                    attr_scores = summarize_attributions(_attr)

                if AttrScoreFunctions.glob_enc_dim_0.value == ExpArgs.attr_score_function:
                    _attr = self.run_glob_enc(txt, input_ids, attention_mask)
                    attr_scores = summarize_attributions(_attr.squeeze(), sum_dim = 0)

                if AttrScoreFunctions.llm.value == ExpArgs.attr_score_function:
                    self.model = self.model.to("cpu")
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.instruct_model = self.instruct_model.to("cuda")
                    try:
                        with torch.no_grad():
                            prompt = "\n\n".join([
                                f"Sort the words of the given text by its importance for {self.task.task_definition} task, with comma between the words. Write END! at the end. Start the answer with Words:. Write nothing else.",
                                "\n".join([f"Text: {txt}", "Words:"])])
                            prompt_input = self.instruct_tokenizer.encode(prompt, return_tensors = "pt")
                            out = self.instruct_model.generate(prompt_input.cuda(), top_p = 0.0, temperature = 1e-10,
                                                               max_new_tokens = len(txt.split()) + 80)
                            out = out[:, prompt_input.shape[-1]:]
                            keywords = self.tokenizer.batch_decode(out)[0]
                            keywords = keywords.replace(self.instruct_tokenizer.eos_token, "")
                            valid_keywords = []
                            for k in keywords.split():
                                w = k.strip()
                                w = w.replace(",", "")
                                if w.lower() == "end!":
                                    break
                                elif len(w) < 2:
                                    continue
                                if (w[0] == "_") or (w[0] == "-"):
                                    w = w[1:]
                                elif w[0].isdigit() and w[1] == ".":
                                    continue
                                elif w[0].isdigit() and w[1].isdigit():
                                    continue
                                else:
                                    valid_keywords.append(w)
                            keywords = valid_keywords
                            print(f"text-{txt}", flush = True)
                            print(f"keywords-{keywords}", flush = True)
                            origin_input_ids_squeezed = origin_input_ids.squeeze()
                            attr_scores = torch.zeros_like(origin_input_ids, dtype = torch.float64).squeeze()
                            if len(keywords) > 0:
                                for idx_k, k in enumerate(keywords):
                                    try:
                                        for token in self.tokenizer.encode(k, add_special_tokens = False):
                                            indices = torch.nonzero(
                                                torch.eq(origin_input_ids_squeezed, token)).squeeze()
                                            try:
                                                attr_scores[indices] += 1 - (idx_k / len(keywords))
                                            except Exception as e2:
                                                print(f"issue - indices - {indices}. e2: {e2}", flush = True)
                                    except Exception as e1:
                                        print(f"issue - for token in self.tokenizer.encode - {e1}", flush = True)

                            print(f"attr_scores-{attr_scores}", flush = True)
                    except Exception as e:
                        print("ERR - {e}", flush = True)
                        attr_scores = torch.ones_like(origin_input_ids.squeeze(), dtype = torch.float64)
                        print(f"4-attr_scores-{attr_scores}", flush = True)
                    self.instruct_model = self.instruct_model.to("cpu")
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.model = self.model.to("cuda")

                if attr_scores is None:
                    raise ValueError("attr_scores score can not be none")

                gc.collect()
                torch.cuda.empty_cache()

                attr_scores = attr_scores.squeeze()
                input_ids = input_ids.squeeze()
                top_items = [self.tokenizer.convert_ids_to_tokens(t) for t in
                             input_ids.squeeze()[attr_scores.topk(min(attr_scores.shape[-1]-1, 10)).indices.unsqueeze(0).tolist()].tolist()]
                df_data[-1][attr_score_function] = top_items

                print(f"{ExpArgs.attr_score_function} - {top_items}")

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

    def run_glob_enc(self, txt, input_ids, attention_mask):
        if self.glob_enc is None:
            self.glob_enc = self.glob_enc_baseline(self.model_path, self.model, self.task)
        _attr = self.glob_enc.compute_attr(txt, input_ids, attention_mask)
        _attr = torch.tensor(_attr)
        return _attr

    def solvability_func(self, sentences):
        sentences = [' '.join(s) for s in sentences]
        default_val = self.tokenizer.cls_token if is_model_encoder_only() else self.tokenizer.eos_token
        sentences = [default_val if not sent else sent for sent in sentences]
        tok = self.tokenizer(sentences, return_tensors = 'pt', padding = True, add_special_tokens = False).to(
            self.device)
        with torch.no_grad():
            # logits = self.model(**tok)['logits']
            logits = run_model(model = self.model, input_ids = tok.input_ids, attention_mask = tok.attention_mask,
                               is_return_logits = True)

        probs = torch.nn.functional.softmax(logits, dim = -1).cpu().numpy()
        return probs

    def lime_func(self, sentences):
        with torch.no_grad():
            logits = run_model(model = self.model, input_ids = sentences, is_return_logits = True)

        probs = torch.nn.functional.softmax(logits, dim = -1).cpu()
        return probs

    def get_alti_input(self, txt):
        alti_max_len = 350
        if ExpArgs.task.name in [IMDB_TASK.name]:
            alti_max_len = 310
        if ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
            self.tokenizer.model_max_length = alti_max_len
        elif ExpArgs.explained_model_backbone == ModelBackboneTypes.MISTRAL.value:
            self.tokenizer.model_max_length = alti_max_len
        else:
            raise ValueError(f"explained_model_backbone issue")

        # max_shots = 1
        # prompt_shots = self.task.llm_few_shots[
        #                :max_shots] if max_shots != -1 else self.task.llm_few_shots
        # alti_llm_few_shots_prompt = "\n\n".join(
        #      ["\n".join([TEXT_PROMPT + i[0], LABEL_PROMPT + str(i[1])]) for i in prompt_shots])
        # alti_task_prompt = "\n\n".join([self.task.llm_task_prompt, alti_llm_few_shots_prompt, TEXT_PROMPT])

        alti_task_prompt = "\n\n".join([self.task.llm_task_prompt, TEXT_PROMPT])
        alti_task_prompt_input_ids, alti_task_prompt_attention_mask = self.encode(alti_task_prompt, True)

        (alti_input_ids, _, _, _, _, _, _, _, alti_attention_mask) = get_inputs(tokenizer = self.tokenizer,
                                                                                model = self.model,
                                                                                model_name = self.model_name,
                                                                                ref_token = self.ref_token, text = txt,
                                                                                device = self.device)
        alti_input_ids, alti_attention_mask = merge_prompts(inputs = alti_input_ids,
                                                            attention_mask = alti_attention_mask,
                                                            task_prompt_input_ids = alti_task_prompt_input_ids,
                                                            label_prompt_input_ids = self.label_prompt_input_ids,
                                                            task_prompt_attention_mask = alti_task_prompt_attention_mask,
                                                            label_prompt_attention_mask = self.label_prompt_attention_mask)

        self.task_prompt_input_ids = alti_task_prompt_input_ids
        self.task_prompt_attention_mask = alti_task_prompt_attention_mask

        return alti_input_ids, alti_attention_mask
