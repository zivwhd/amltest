from typing import Union, Dict, List, Tuple

from config.constants import TEXT_PROMPT, LABEL_PROMPT, WORDS


class Task:
    def __init__(self, dataset_name: str, dataset_train: str, dataset_val: str, dataset_test: str,
                 dataset_column_text: str, dataset_column_label: str, bert_fine_tuned_model: str,
                 roberta_fine_tuned_model: str, distilbert_fine_tuned_model: str, roberta_base_model: str,
                 distilbert_base_model: str, bert_base_model: str, llama_model: str,
                 labels_str_int_maps: Union[Dict, None], default_lr: float, llama_lr: float,
                 test_sample: Union[int, None], name: str, llama_task_prompt: str,
                 llama_few_shots_prompt: List[Tuple[str, int]], llama_is_for_seq_class = False, llama_max_length:Union[int,None]=None,
                 peft_llama_adapter = None, baseline_llama_few_shots_prompt: List[Tuple[str, int, List[str]]] = []):
        self.dataset_name = dataset_name
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.dataset_column_text = dataset_column_text
        self.dataset_column_label = dataset_column_label
        self.bert_fine_tuned_model = bert_fine_tuned_model
        self.roberta_fine_tuned_model = roberta_fine_tuned_model
        self.distilbert_fine_tuned_model = distilbert_fine_tuned_model

        self.roberta_base_model = roberta_base_model
        self.distilbert_base_model = distilbert_base_model
        self.bert_base_model = bert_base_model

        self.llama_model = llama_model
        self.labels_str_int_maps = labels_str_int_maps
        self.labels_int_str_maps = {value: key for key, value in
                                    labels_str_int_maps.items()} if labels_str_int_maps else None
        self.default_lr = default_lr
        self.llama_lr = llama_lr
        self.test_sample = test_sample
        self.name = name
        self.llama_is_for_seq_class = llama_is_for_seq_class
        self.llama_max_length = llama_max_length
        self.peft_llama_adapter = peft_llama_adapter
        self.llama_task_prompt = llama_task_prompt
        self.llama_few_shots = llama_few_shots_prompt  # for test only
        self.llama_few_shots_prompt = "\n\n".join(
            ["\n".join([TEXT_PROMPT + i[0], LABEL_PROMPT + str(i[1])]) for i in llama_few_shots_prompt])
        self.baseline_llama_few_shots_prompt = baseline_llama_few_shots_prompt
        self.baseline_llama_few_shots_prompt_str = "\n\n".join(  #
            ["You will be provided with a block of text, and your task is to extract a list of keywords from it."] +  #
            [  #
                "\n".join([TEXT_PROMPT + i[0], LABEL_PROMPT + str(i[1]), WORDS + ",".join([str(n) for n in i[2]])]) for
                i in baseline_llama_few_shots_prompt  #
            ]  #
        )
