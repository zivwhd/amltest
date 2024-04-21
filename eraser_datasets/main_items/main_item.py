import os
import os
import pickle
import sys
from typing import List

import pandas as pd

from eraser_datasets.eraser_utils import scores_per_word_from_scores_per_token
from main.utils.baselines_utils import get_model, get_data, get_tokenizer, init_baseline_exp
from main.utils.baslines_model_functions import ForwardModel, get_inputs
from main.utils.seg_ig import SequentialIntegratedGradients

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from config.config import BackbonesMetaData, ExpArgs
from config.types_enums import RefTokenNameTypes, AttrScoreFunctions, EvalMetric
from utils.utils_functions import (run_model, get_device, is_model_encoder_only)
from eraser_datasets.eraser_metrics import main
import torch

from captum.attr import (DeepLift, GradientShap, InputXGradient, IntegratedGradients, Lime)


def summarize_attributions(attributions, sum_dim = -1):
    attributions = attributions.sum(dim = sum_dim).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


class BaselinesItem:
    def __init__(self, attr_score_function: str, data, metrics: List[str] = None):
        print(f"run {attr_score_function}")
        init_baseline_exp()
        self.task = ExpArgs.task
        self.metrics = metrics if metrics is not None else ["comp_suff"]
        self.data = data
        self.model, self.model_path = get_model()
        self.tokenizer = get_tokenizer(self.model_path)
        self.model_name = BackbonesMetaData.name[ExpArgs.explained_model_backbone]
        ExpArgs.attr_score_function = attr_score_function
        self.glob_enc = None

        self.output_path = "./OUTPUT"
        os.makedirs(self.output_path, exist_ok = True)
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

        if ExpArgs.attr_score_function == AttrScoreFunctions.decompX.value:
            sys.path.append(f"{os.getcwd()}/../main/utils/DecompX")
            from main.utils.decompX_utils import DecomposeXBaseline
        elif ExpArgs.attr_score_function == AttrScoreFunctions.alti.value:
            sys.path.append(f"{os.getcwd()}/../main/utils/transformer-contributions/alti")
            from main.utils.alti_utils import AltiBaseline
        elif (ExpArgs.attr_score_function == AttrScoreFunctions.glob_enc.value) or (
                ExpArgs.attr_score_function == AttrScoreFunctions.glob_enc_dim_0.value):
            sys.path.append(f"{os.getcwd()}/../main/utils/GlobEnc")
            from main.utils.globenc_utils import GlobEncBaseline
            self.glob_enc_baseline = GlobEncBaseline
        elif ExpArgs.attr_score_function == AttrScoreFunctions.solvability.value:
            sys.path.append(f"{os.getcwd()}/../main/utils/solvability-explainer")
            if not is_model_encoder_only():
                self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
                self.tokenizer.padding_side = "left"
            from solvex import BeamSearchExplainer, TextWordMasker

        # Prepare forward model
        nn_forward_func = ForwardModel(model = self.model, model_name = self.model_name)
        # Compute attributions

        gt = []
        preds = []

        all_data = []
        for index, row in self.data.iterrows():
            txt = row["document"]
            (input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed,
             ref_type_embed, attention_mask) = get_inputs(tokenizer = self.tokenizer, model = self.model,
                                                          model_name = self.model_name, ref_token = self.ref_token,
                                                          text = txt, device = self.device)

            with torch.no_grad():
                pred_origin_logits = run_model(model = self.model, input_ids = input_ids,
                                               attention_mask = attention_mask, is_return_logits = True)
                model_pred_origin = torch.argmax(pred_origin_logits, dim = 1)
            self.model.zero_grad()

            attr_scores = None

            if AttrScoreFunctions.deep_lift.value == ExpArgs.attr_score_function:
                explainer = DeepLift(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = ref_input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr_scores = summarize_attributions(_attr)

            if AttrScoreFunctions.gradient_shap.value == ExpArgs.attr_score_function:
                explainer = GradientShap(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = torch.cat([ref_input_embed, input_embed]),
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr_scores = summarize_attributions(_attr)

            if AttrScoreFunctions.lime.value == ExpArgs.attr_score_function:
                explainer = Lime(self.lime_func)
                _attr = explainer.attribute(input_ids)
                attr_scores = _attr.squeeze().detach()

            if AttrScoreFunctions.input_x_gradient.value == ExpArgs.attr_score_function:
                explainer = InputXGradient(nn_forward_func)
                _attr = explainer.attribute(input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr_scores = summarize_attributions(_attr)

            if AttrScoreFunctions.integrated_gradients.value == ExpArgs.attr_score_function:
                explainer = IntegratedGradients(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = ref_input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr_scores = summarize_attributions(_attr)

            if AttrScoreFunctions.sequential_integrated_gradients.value == ExpArgs.attr_score_function:
                explainer = SequentialIntegratedGradients(nn_forward_func)
                _attr = explainer.attribute(input_embed, baselines = ref_input_embed,
                                            additional_forward_args = (attention_mask, position_embed, type_embed,), )
                attr_scores = summarize_attributions(_attr)

            if AttrScoreFunctions.decompX.value == ExpArgs.attr_score_function:
                decompse = DecomposeXBaseline(self.model_path)
                attr_scores = decompse.compute_attr(input_ids, attention_mask)

            if AttrScoreFunctions.alti.value == ExpArgs.attr_score_function:
                alti = AltiBaseline(self.model)
                _attr = alti.compute_attr(input_ids, attention_mask)
                attr_scores = summarize_attributions(_attr, sum_dim = 0)

            if AttrScoreFunctions.glob_enc.value == ExpArgs.attr_score_function:
                _attr = self.run_glob_enc(txt, input_ids, attention_mask)
                attr_scores = summarize_attributions(_attr)

            if AttrScoreFunctions.glob_enc_dim_0.value == ExpArgs.attr_score_function:
                _attr = self.run_glob_enc(txt, input_ids, attention_mask)
                attr_scores = summarize_attributions(_attr.squeeze(), sum_dim = 0)

            if AttrScoreFunctions.solvability.value == ExpArgs.attr_score_function:
                # sentence = self.tokenizer.tokenize(txt, add_special_tokens = True)
                sentence = [self.tokenizer.convert_ids_to_tokens(i) for i in input_ids.squeeze().tolist()]

                masker = TextWordMasker(suppression = 'remove')
                explainer = BeamSearchExplainer(masker, f = self.solvability_func, beam_size = 10, batch_size = 50,
                                                metric = 'comp-suff')
                e = explainer.explain_instance(sentence, label = model_pred_origin.squeeze().item())
                attr_scores = torch.tensor(e["exp"])

            to_save = row.to_dict()
            to_save["attr_scores"] = attr_scores.detach().cpu()
            all_data.append(dict(to_save))

            with open(f"{self.output_path}/{ExpArgs.attr_score_function}", 'wb') as f:
                pickle.dump(all_data, f)


        #     scores_per_word = scores_per_word_from_scores_per_token(row["document"].split(), self.tokenizer, input_ids,
        #                                                             attr_scores)
        #
        #     hard_rationales = []
        #
        #     tok_k = int(0.07 * attr_scores.shape[-1])
        #     print("calculating top ", tok_k)
        #     _, indices = scores_per_word.topk(k = tok_k)
        #     for index in indices.tolist():
        #         hard_rationales.append({"start_token": index, "end_token": index + 1})
        #     pred = {"annotation_id": row["annotation_id"],
        #             "rationales": [{"docid": row["annotation_id"], "hard_rationale_predictions": hard_rationales}], }
        #     preds.append(pred)
        #     gt.append(row)
        #
        # scores = main(preds, gt)
        # # Open the file in binary write mode
        # with open(f"{self.output_path}/{ExpArgs.attr_score_function}.pkl", 'wb') as f:
        #     # Dump the data into the file using pickle
        #     pickle.dump(scores, f)

        a = 1

    def run_glob_enc(self, txt, input_ids, attention_mask):
        if self.glob_enc is None:
            self.glob_enc = self.glob_enc_baseline(self.model_path, self.model, self.task)
        _attr = self.glob_enc.compute_attr(txt, input_ids, attention_mask)
        _attr = torch.tensor(_attr)
        return _attr

    def solvability_func(self, sentences):
        print(sentences)
        sentences = [' '.join(s) for s in sentences]
        default_val = self.tokenizer.cls_token if is_model_encoder_only() else self.tokenizer.eos_token
        sentences = [default_val if not sent else sent for sent in sentences]
        tok = self.tokenizer(sentences, return_tensors = 'pt', truncation = True, padding = True,
                             add_special_tokens = False).to(self.device)
        with torch.no_grad():
            # logits = self.model(**tok)['logits']
            logits = run_model(model = self.model, input_ids = tok.input_ids, attention_mask = tok.attention_mask,
                               is_return_logits = True)

        probs = torch.nn.functional.softmax(logits, dim = -1).cpu().numpy()
        return probs

    def lime_func(self, sentences):
        with torch.no_grad():
            logits = run_model(model = self.model, input_ids = sentences, is_return_logits = True)

        probs = torch.nn.functional.softmax(logits, dim = -1).squeeze().cpu().numpy()
        return probs
