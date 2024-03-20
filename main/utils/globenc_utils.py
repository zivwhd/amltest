import numpy as np
import torch
from GlobEnc.src.attention_rollout import AttentionRollout
from GlobEnc.src.modeling.modeling_bert import BertForSequenceClassification
from GlobEnc.src.modeling.new_modeling_roberta import RobertaForSequenceClassification

from config.config import ExpArgs
from config.types_enums import ModelBackboneTypes


# based on https://github.com/mt-upc/transformer-contributions/blob/main/Text_classification.ipynb
class GlobEncBaseline:

    def __init__(self, model_path, model, task):
        self.model_path = model_path
        self.model = self.load_model(model)
        self.model.cuda()
        self.task = task

    def load_model(self, model):
        # if "bert" in self.model_path:
        if ExpArgs.explained_model_backbone == ModelBackboneTypes.BERT.value:
            model = BertForSequenceClassification.from_pretrained(self.model_path)
            return model
        elif ExpArgs.explained_model_backbone == ModelBackboneTypes.ROBERTA.value:
            model = RobertaForSequenceClassification.from_pretrained(self.model_path)
            return model
        # elif "electra" in self.model_path:
        #     return ElectraForSequenceClassification.from_pretrained(self.model_path)
        else:
            return model

    def compute_attr(self, txt, input_ids, attention_mask):
        with torch.no_grad():
            logits, attentions, norms = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                                   output_attentions=True, output_norms=True, return_dict=False)
            num_layers = len(attentions)
            norm_nenc = torch.stack([norms[i][4] for i in range(num_layers)]).squeeze().cpu().numpy()
            print("Single layer N-Enc token attribution:", norm_nenc.shape)

            # Aggregate and compute GlobEnc
            globenc = AttentionRollout().compute_flows([norm_nenc], output_hidden_states=False)[0]
            globenc = np.array(globenc)
            print("Aggregated N-Enc token attribution (GlobEnc):", globenc.shape)
            return globenc
