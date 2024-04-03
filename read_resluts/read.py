import pandas as pd
from glob import glob

url = "/home/yonatanto/work/theza/EXPLAINABILITY/EXP_BASELINES/V002/runs/baselines/OUT/*/*/results.csv"
files = glob(url)

def reaad_file(f):
    file_df = pd.read_csv(f)
    file_df["folder"] = "_".join(f.split("/")[:-1])
    return file_df

df = pd.concat([reaad_file(f) for f in files])

df.groupby(['task', 'folder', "explained_model_backbone", 'eval_tokens', 'eval_metric'])["metric_result"].mean().to_csv("./results.csv")
