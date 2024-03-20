import numpy as np
import torch
from DecompX.src.decompx_utils import DecompXConfig
from DecompX.src.modeling_bert import BertForSequenceClassification
from DecompX.src.modeling_roberta import RobertaForSequenceClassification

from config.constants import HF_CACHE


class DecomposeXBaseline:

    def __init__(self, model_path):
        self.model_path = model_path  # Only BERT or RoBERTa
        self.CONFIGS = {
            "DecompX": DecompXConfig(include_biases=True, bias_decomp_type="absdot", include_LN1=True, include_FFN=True,
                                     FFN_approx_type="GeLU_ZO", include_LN2=True, aggregation="vector",
                                     include_classifier_w_pooler=True, tanh_approx_type="ZO", output_all_layers=True,
                                     output_attention=None, output_res1=None, output_LN1=None, output_FFN=None,
                                     output_res2=None, output_encoder=None, output_aggregated="norm",
                                     output_pooler="norm", output_classifier=True, ), }
        self.model = self.get_model()
        self.model.eval()
        self.model.cuda()

    def get_model(self):
        if "roberta" in self.model_path:
            return RobertaForSequenceClassification.from_pretrained(self.model_path, cache_dir=HF_CACHE)
        elif "bert" in self.model_path:
            return BertForSequenceClassification.from_pretrained(self.model_path, cache_dir=HF_CACHE)
        else:
            raise Exception(f"Not implented model: {self.model_path}")

    def compute_attr(self, input_ids, attention_mask):
        with torch.no_grad():
            batch_lengths = attention_mask.sum(dim=-1)
            logits, hidden_states, decompx_last_layer_outputs, decompx_all_layers_outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, output_attentions=False, return_dict=False,
                output_hidden_states=True, decompx_config=self.CONFIGS["DecompX"])

        importance = np.array([g.squeeze().cpu().detach().numpy() for g in
                               decompx_last_layer_outputs.aggregated]).squeeze()

        # importance = [importance[:batch_lengths, :batch_lengths] for j in range(len(importance))]
        importance = torch.tensor(importance)[0, :] # aggregated
        importance = importance / np.abs(importance).max() / 1.5  # Normalize

        return importance
