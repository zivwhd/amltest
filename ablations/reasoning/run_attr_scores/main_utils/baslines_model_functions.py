import torch
import torch.nn as nn

from config.config import BackbonesMetaData, ExpArgs
from config.types_enums import ModelBackboneTypes
from utils.utils_functions import run_model, is_model_encoder_only


class ForwardModel(nn.Module):
    def __init__(self, model, model_name):
        super().__init__()
        self.model = model
        self.model_name = model_name

    def forward(self, input_embed, attention_mask = None, position_embed = None, type_embed = None,
                return_all_logits = False):

        if is_model_encoder_only():
            embeds = input_embed + position_embed
            if type_embed is not None:
                embeds += type_embed

            # Get predictions
            embeds = getattr(self.model, self.model_name).embeddings.dropout(
                getattr(self.model, self.model_name).embeddings.LayerNorm(embeds))
            pred = run_model(model = self.model, inputs_embeds = embeds, attention_mask = attention_mask,
                             is_return_logits = True)
        else:
            pred = run_model(model = self.model, inputs_embeds = input_embed, attention_mask = attention_mask,
                             is_return_logits = True)

        # Return all logits or just maximum class
        if return_all_logits:
            return pred
        else:
            return pred.max(1).values


def construct_input_ref_pair(tokenizer, text, ref_token_id, sep_token_id, cls_token_id, device):
    text_ids = tokenizer.encode(text, add_special_tokens = False, truncation = True,
                                max_length = tokenizer.max_len_single_sentence, )
    input_ids = ([cls_token_id] + text_ids + [sep_token_id])  # construct input token ids
    ref_input_ids = ([cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id])  # construct reference token ids

    return torch.tensor([input_ids], device = device), torch.tensor([ref_input_ids], device = device)


def construct_input_ref_pos_id_pair(model, model_name, input_ids, device):
    seq_length = input_ids.size(1)

    if ExpArgs.explained_model_backbone == ModelBackboneTypes.DISTILBERT.value:
        position_ids = torch.arange(seq_length, dtype = torch.long, device = device)
        ref_position_ids = torch.arange(seq_length, dtype = torch.long, device = device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)

    else:
        position_ids = (getattr(model, model_name).embeddings.position_ids[:, 0:seq_length].to(device))
        ref_position_ids = (getattr(model, model_name).embeddings.position_ids[:, 0:seq_length].to(device))

    return position_ids, ref_position_ids


def construct_input_ref_token_type_pair(input_ids, device):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0] * seq_len], dtype = torch.long, device = device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, dtype = torch.long, device = device)
    return token_type_ids, ref_token_type_ids


def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)


def get_word_embeddings(model):
    model_name = BackbonesMetaData.name[ExpArgs.explained_model_backbone]
    if is_model_encoder_only():
        return getattr(model, model_name).embeddings.word_embeddings.weight
    else:
        return getattr(model, model_name).get_input_embeddings().weight


def construct_word_embedding(model, model_name, input_ids):
    return getattr(model, model_name).embeddings.word_embeddings(input_ids)


def construct_position_embedding(model, model_name, position_ids):
    return getattr(model, model_name).embeddings.position_embeddings(position_ids)


def construct_type_embedding(model, model_name, type_ids):
    return getattr(model, model_name).embeddings.token_type_embeddings(type_ids)


def construct_sub_embedding(model, model_name, input_ids, ref_input_ids, position_ids, ref_position_ids, type_ids,
                            ref_type_ids, ):
    input_embeddings = construct_word_embedding(model, model_name, input_ids)
    ref_input_embeddings = construct_word_embedding(model, model_name, ref_input_ids)
    input_position_embeddings = construct_position_embedding(model, model_name, position_ids)
    ref_input_position_embeddings = construct_position_embedding(model, model_name, ref_position_ids)

    if type_ids is not None:
        input_type_embeddings = construct_type_embedding(model, model_name, type_ids)
    else:
        input_type_embeddings = None
    if ref_type_ids is not None:
        ref_input_type_embeddings = construct_type_embedding(model, model_name, ref_type_ids)
    else:
        ref_input_type_embeddings = None

    return ((input_embeddings, ref_input_embeddings), (input_position_embeddings, ref_input_position_embeddings),
            (input_type_embeddings, ref_input_type_embeddings),)


def get_base_token_emb(model, model_name, ref_token, device):
    return construct_word_embedding(model, model_name, torch.tensor([ref_token], device = device), )


def get_tokens(tokenizer, text_ids):
    return tokenizer.convert_ids_to_tokens(text_ids.squeeze())


def get_inputs(tokenizer, model, model_name, ref_token, text, device):
    if is_model_encoder_only():
        return get_inputs_encoder_only(tokenizer, model, model_name, ref_token, text, device)
    else:
        return get_inputs_llm(tokenizer, model, model_name, ref_token, text, device)


def get_inputs_encoder_only(tokenizer, model, model_name, ref_token, text, device):
    ref_token_id = ref_token
    sep_token_id = tokenizer.sep_token_id
    cls_token_id = tokenizer.cls_token_id

    input_ids, ref_input_ids = construct_input_ref_pair(tokenizer, text, ref_token_id, sep_token_id, cls_token_id,
                                                        device)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(model, model_name, input_ids, device)

    if ExpArgs.explained_model_backbone == ModelBackboneTypes.DISTILBERT.value:
        type_ids, ref_type_ids = None, None
    else:
        type_ids, ref_type_ids = construct_input_ref_token_type_pair(input_ids, device)
    attention_mask = construct_attention_mask(input_ids)

    ((input_embed, ref_input_embed), (position_embed, ref_position_embed),
     (type_embed, ref_type_embed),) = construct_sub_embedding(model, model_name, input_ids, ref_input_ids, position_ids,
                                                              ref_position_ids, type_ids, ref_type_ids, )

    return (input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed,
            ref_type_embed, attention_mask,)


def get_inputs_llm(tokenizer, model, model_name, ref_token, text, device):
    ref_token_id = ref_token
    # sep_token_id = tokenizer.encode(tokenizer.special_tokens_map["eos_token"])[0]

    if ExpArgs.task.is_finetuned_with_lora:
        text_ids = tokenizer.encode(text, add_special_tokens = True, truncation = True)
    else:
        text_ids = tokenizer.encode(text, add_special_tokens = False, truncation = True)
    input_ids = torch.tensor(text_ids, device = device).unsqueeze(0)  # construct input token ids
    ref_input_ids = torch.tensor(([ref_token_id] * len(text_ids)), device = device).unsqueeze(
        0)  # construct reference token ids

    input_embed = model.get_input_embeddings()(input_ids)
    ref_input_embed = model.get_input_embeddings()(ref_input_ids)

    attention_mask = construct_attention_mask(input_ids)

    position_embed = torch.tensor([])
    ref_position_embed = torch.tensor([])
    type_embed = torch.tensor([])
    ref_type_embed = torch.tensor([])

    return (input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed,
            ref_type_embed, attention_mask,)
