import gc
import os
import sys
import time, datetime
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.tasks import IMDB_TASK, AGN_TASK, RTN_TASK, SST_TASK, EMOTION_TASK
from main.utils.baselines_utils import get_model, get_data, get_tokenizer, init_baseline_exp
from main.utils.baslines_model_functions import ForwardModel, get_inputs
from main.utils.seg_ig import SequentialIntegratedGradients
from utils.dataclasses.evaluations import DataForEvaluation, DataForEvaluationInputs

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from config.config import BackbonesMetaData, ExpArgs
from config.constants import (TEXT_PROMPT, LABEL_PROMPT_NEW_LINE, LOCAL_MODELS_PREFIX, HF_CACHE)
from config.types_enums import RefTokenNameTypes, AttrScoreFunctions, EvalMetric, ModelBackboneTypes
from utils.utils_functions import (run_model, get_device, is_model_encoder_only, merge_prompts, conv_to_word_embedding,
                                   is_use_prompt)

import torch
from main.sloc import Sloc
from captum.attr import (DeepLift, GradientShap, InputXGradient, IntegratedGradients, Lime)
from evaluations.evaluations import evaluate_tokens_attributions


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

        if ExpArgs.attribution_scores_function == AttrScoreFunctions.decompX.value:
            sys.path.append(f"{os.getcwd()}/../../main/utils/DecompX")
            from main.utils.decompX_utils import DecomposeXBaseline
        elif ExpArgs.attribution_scores_function == AttrScoreFunctions.decompX_class.value:
            sys.path.append(f"{os.getcwd()}/../../main/utils/DecompX")
            from main.utils.decompX_utils_class import DecomposeXBaseline
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
                raise ValueError("solvability not working with LLMs")
            from main.utils.solvability_explainer.solvex.explainer import BeamSearchExplainer
            from main.utils.solvability_explainer.solvex.masker import TextWordMasker

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
            # ref_input_ids, _ = self.merge_prompts_handler(origin_ref_input_ids.clone(), attention_mask)
            # input_embed, _ = self.merge_prompts_embeddings__handler(origin_input_embed.clone(), attention_mask)
            # ref_input_embed, _ = self.merge_prompts_embeddings__handler(origin_ref_input_embed.clone(), attention_mask)

            input_ids = origin_input_ids
            input_embed = origin_input_embed
            ref_input_embed = origin_ref_input_embed

            # print(f"merged_input_ids: {merged_input_ids}")
            with torch.no_grad():
                explained_model_logits = run_model(model = self.model, input_ids = merged_input_ids,
                                                   attention_mask = merged_attention_mask, is_return_logits = True)
                explained_model_predicted_class = torch.argmax(explained_model_logits, dim = 1)
            self.model.zero_grad()

            begin = time.time()

            attribution_scores = None

            if ExpArgs.attribution_scores_function == AttrScoreFunctions.deep_lift.value:
                explainer = DeepLift(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = ref_input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attribution_scores = summarize_attributions(_attr)

            if ExpArgs.attribution_scores_function == AttrScoreFunctions.gradient_shap.value:
                explainer = GradientShap(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = torch.cat([ref_input_embed, input_embed]),
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attribution_scores = summarize_attributions(_attr)
            
            if ExpArgs.attribution_scores_function == "TESTSLOC":

                with torch.no_grad():
                    print("input_ids", input_ids)
                    logits = run_model(model = self.model, input_ids = input_ids, is_return_logits = True)
                    #probs = run_model(model = self.model, input_ids = input_ids, is_return_logits = False)
                    print("logits", logits)
                    #print("probs", logits)
                ##
                explainer = Lime(self.lime_func)
                _attr = explainer.attribute(input_ids, target = explained_model_logits.max(1)[1])
                attribution_scores = _attr.squeeze().detach()
                print("attr-scores:", attribution_scores)
                print("attr-scores-shape:", attribution_scores.shape)
                print("logits:", explained_model_logits)
                print("target:", explained_model_logits.max(1)[1])

                #probs = torch.nn.functional.softmax(logits, dim = -1).cpu()         
                raise Exception("Not implemented") 

            if ExpArgs.attribution_scores_function in [AttrScoreFunctions.sloc.value, AttrScoreFunctions.slocB.value]:  
                eval_model = lambda inp: run_model(model = self.model, input_ids = inp, is_return_logits = True)
                with_bias = ExpArgs.attribution_scores_function in [AttrScoreFunctions.slocB.value]
                explainer = Sloc(with_bias=with_bias)
                attribution_scores = explainer.run(eval_model, input_ids, target = explained_model_logits.max(1)[1])

            if ExpArgs.attribution_scores_function == AttrScoreFunctions.lime.value:
                explainer = Lime(self.lime_func)
                _attr = explainer.attribute(input_ids, target = explained_model_logits.max(1)[1])
                attribution_scores = _attr.squeeze().detach()

            if ExpArgs.attribution_scores_function == AttrScoreFunctions.input_x_gradient.value:
                explainer = InputXGradient(nn_forward_func)
                _attr = explainer.attribute(input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attribution_scores = summarize_attributions(_attr)

            if ExpArgs.attribution_scores_function == AttrScoreFunctions.integrated_gradients.value:
                explainer = IntegratedGradients(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = ref_input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attribution_scores = summarize_attributions(_attr)

            if ExpArgs.attribution_scores_function == AttrScoreFunctions.sequential_integrated_gradients.value:
                explainer = SequentialIntegratedGradients(nn_forward_func)

                n_steps = 50  # default value

                if is_model_encoder_only():
                    # print("C"*100)
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

            if ExpArgs.attribution_scores_function == AttrScoreFunctions.decompX.value:
                decompse = DecomposeXBaseline(self.model_path)
                attribution_scores = decompse.compute_attr(input_ids, attention_mask)
                attribution_scores = torch.tensor(attribution_scores)

            if ExpArgs.attribution_scores_function == AttrScoreFunctions.decompX_class.value:
                decompse = DecomposeXBaseline(self.model_path)
                attribution_scores = decompse.compute_attr(input_ids, attention_mask, explained_model_predicted_class)
                attribution_scores = torch.tensor(attribution_scores)

            if ExpArgs.attribution_scores_function == AttrScoreFunctions.alti.value:
                alti_input_ids, alti_attention_mask = input_ids.clone(), attention_mask.clone()
                origin_model_max_length = self.tokenizer.model_max_length
                if (not is_model_encoder_only()) and (self.task.name in [IMDB_TASK.name, AGN_TASK.name]):
                    alti_input_ids, alti_attention_mask = self.get_alti_input(txt)

                alti = AltiBaseline(self.model)
                attribution_scores = alti.compute_attr(alti_input_ids, alti_attention_mask)

                del alti

                if origin_model_max_length != self.tokenizer.model_max_length:
                    self.tokenizer.model_max_length = origin_model_max_length

            if ExpArgs.attribution_scores_function == AttrScoreFunctions.glob_enc.value:
                attribution_scores = self.run_glob_enc(txt, input_ids, attention_mask)

            if ExpArgs.attribution_scores_function == AttrScoreFunctions.solvability.value:
                sentence = [self.tokenizer.convert_ids_to_tokens(i) for i in input_ids.squeeze().tolist()]

                if len(self.metrics) != 1:
                    raise ValueError("Err")
                if self.metrics[0].value in [EvalMetric.COMPREHENSIVENESS.value,
                                             EvalMetric.AOPC_COMPREHENSIVENESS.value]:
                    metric = 'comp'
                    suppression = 'remove'
                elif self.metrics[0].value in [EvalMetric.SUFFICIENCY.value, EvalMetric.AOPC_SUFFICIENCY.value, ]:
                    metric = 'suff'
                    suppression = 'remove'
                elif self.metrics[0].value in [EvalMetric.EVAL_LOG_ODDS.value]:
                    metric = 'comp'  # metric = 'comp-suff'
                    suppression = f'replace-{self.tokenizer.convert_ids_to_tokens(self.ref_token)}'
                else:
                    raise ValueError("Err")

                masker = TextWordMasker(suppression = suppression)
                explainer = BeamSearchExplainer(masker, f = self.solvability_func,
                                                beam_size = ExpArgs.SOLVABILITY_BATCH_SIZE, batch_size = 50,
                                                metric = metric)
                e = explainer.explain_instance(sentence, label = explained_model_predicted_class.squeeze().item())
                attribution_scores = torch.tensor(e["exp"])

            if ExpArgs.attribution_scores_function == AttrScoreFunctions.llm.value:
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
                        attribution_scores = torch.zeros_like(origin_input_ids, dtype = torch.float64).squeeze()
                        if len(keywords) > 0:
                            for idx_k, k in enumerate(keywords):
                                try:
                                    for token in self.tokenizer.encode(k, add_special_tokens = False):
                                        indices = torch.nonzero(torch.eq(origin_input_ids_squeezed, token)).squeeze()
                                        try:
                                            attribution_scores[indices] += 1 - (idx_k / len(keywords))
                                        except Exception as e2:
                                            print(f"issue - indices - {indices}. e2: {e2}", flush = True)
                                except Exception as e1:
                                    print(f"issue - for token in self.tokenizer.encode - {e1}", flush = True)

                        print(f"attribution_scores-{attribution_scores}", flush = True)
                except Exception as e:
                    print("ERR - {e}", flush = True)
                    attribution_scores = torch.ones_like(origin_input_ids.squeeze(), dtype = torch.float64)
                self.instruct_model = self.instruct_model.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
                self.model = self.model.to("cuda")

            if attribution_scores is None:
                raise ValueError("attribution_scores score can not be none")

            gc.collect()
            torch.cuda.empty_cache()

            eval_attr_score = attribution_scores

            if ExpArgs.is_evaluate:
                # if is_use_prompt() and (AttrScoreFunctions.llm.value != ExpArgs.attribution_scores_function):
                #     eval_attr_score = attribution_scores[
                #                       self.task_prompt_input_ids.shape[-1]:-self.label_prompt_input_ids.shape[
                #                           -1]].detach()

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
        ##################
        if False:
            print(f"saving - summary")
            input_path = Path(experiment_path) / "results.csv"
            output_path = (Path(experiment_path) / "../../summary.csv").resolve()
            # Read input DataFrame
            df = pd.read_csv(input_path)
            # Group and compute mean
            grouped = (
                df.groupby(['task', 'evaluation_metric', 'explained_model_backbone'])['metric_result']
                .mean()
                .reset_index()
                .rename(columns={'metric_result': 'mean_metric_result'})
            )
            current_time = datetime.now().isoformat(timespec='seconds')  # e.g., '2025-10-07T21:42:00'
            grouped['timestamp'] = current_time
            # Check if summary file exists
            write_header = not output_path.exists()
            # Append to the output file
            grouped.to_csv(output_path, mode='a', index=False, header=write_header)
            print(f"done saving summary")
        ########################
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
