import pandas as pd
from glob import glob

url = "/home/yonatanto/work/theza/EXPLAINABILITY/EXP_BASELINES/V002/runs/baselines/OUT/*/*/results.csv"
files = glob(url)

def reaad_file(f):
    file_df = pd.read_csv(f)
    file_df["folder"] = "_".join(f.split("/")[:-1])
    return file_df

df = pd.concat([reaad_file(f) for f in files])

x = df.groupby(['task', 'folder', "explained_model_backbone", 'eval_tokens', 'eval_metric']).aggregate({"metric_result": ["mean", "count"]}).reset_index()
y = x[(x["task"]=="emotions") & (x["explained_model_backbone"]=="BERT")]
z = y[y["eval_metric"]=="COMPREHENSIVENESS"]