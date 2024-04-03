import pandas as pd

from glob import glob
movies = glob("/home/yonatanto/Downloads/multirc/*.jsonl")

data = [pd.read_json(f, lines=True) for f in movies]

# Open the file in read mode
with open('/home/yonatanto/Downloads/multirc/docs/Fiction-stories-masc-The_Black_Willow-0.txt', 'r') as file:
    doc_1 = file.read()

