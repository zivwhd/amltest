import torch
from solvex import BeamSearchExplainer, TextWordMasker
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config.constants import HF_CACHE
from datasets import load_dataset
from utils.dataclasses import Task


class TestTime:

    def __init__(self, task: Task, beam_size: int):
        self.task = task
        self.beam_size = beam_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        name = task.roberta_fine_tuned_model
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, cache_dir = HF_CACHE).to(
            self.device).eval()

        self.datasets = load_dataset(task.dataset_name)

    def model_func(self, sentences):
        sentences = [' '.join(s) for s in sentences]
        tok = self.tokenizer(sentences, return_tensors = 'pt', padding = True).to(self.device)
        with torch.no_grad():
            logits = self.model(**tok)['logits']
        probs = torch.nn.functional.softmax(logits, dim = -1).cpu().numpy()
        return probs

    def run(self):
        sentence = self.datasets["train"][0][self.task.dataset_column_text].split(' ')
        print(f"{self.task.name} - beam_size: {self.beam_size}. len - {len(sentence)}")
        masker = TextWordMasker(suppression = 'remove')

        import time

        start = time.time()

        explainer = BeamSearchExplainer(masker, f = self.model_func, beam_size = self.beam_size, batch_size = 50, metric = "comp")
        e = explainer.explain_instance(sentence, label = 1)
        print(e)

        end = time.time()
        print(f"executed time: {end - start}")
