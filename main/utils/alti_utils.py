import random

import torch
from alti.contributions import ModelWrapper
from alti.utils_contributions import *


# based on https://github.com/mt-upc/transformer-contributions/blob/main/Text_classification.ipynb
class AltiBaseline:

    def __init__(self, model):
        self.model = model  # Only BERT or RoBERTa and DistilBERT
        random.seed(10)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_wrapped = ModelWrapper(model)


    def compute_attr(self, input_ids, attention_mask):
        pd_batch = dict(input_ids=input_ids, attention_mask=attention_mask)
        prediction_scores, hidden_states, attentions, contributions_data = self.model_wrapped(pd_batch)

        resultant_norm = torch.norm(torch.squeeze(contributions_data['resultants']), p=1, dim=-1)
        normalized_contributions = normalize_contributions(contributions_data['contributions'], scaling='min_sum',
                                                           resultant_norm=resultant_norm)
        contributions_mix = compute_joint_attention(normalized_contributions)
        joint_attention_layer = -1
        pos = 0
        # contributions_mix_cls = contributions_mix[joint_attention_layer][pos]
        contributions_mix_cls = contributions_mix[joint_attention_layer]
        return contributions_mix_cls
