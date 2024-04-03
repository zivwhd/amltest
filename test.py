import pandas as pd

from glob import glob
movies = glob("/home/yonatanto/Downloads/movies/*.jsonl")

data = [pd.read_json(f, lines=True) for f in movies]

# Open the file in read mode
with open('/home/yonatanto/Downloads/movies/docs/negR_000.txt', 'r') as file:
    doc_1 = file.read()

