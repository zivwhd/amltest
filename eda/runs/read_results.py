import pandas as pd
import ast

urls = ['/home/yonatanto/work/theza/EMNLP_EXPLAINABILITY_V4/FEW_SHOTS/EXP_BASELINES/eda/runs/EVAL_imdb_MISTRAL/eval.csv',
 '/home/yonatanto/work/theza/EMNLP_EXPLAINABILITY_V4/FEW_SHOTS/EXP_BASELINES/eda/runs/EVAL_sst_LLAMA/eval.csv',
 '/home/yonatanto/work/theza/EMNLP_EXPLAINABILITY_V4/FEW_SHOTS/EXP_BASELINES/eda/runs/EVAL_sst_MISTRAL/eval.csv',
 '/home/yonatanto/work/theza/EMNLP_EXPLAINABILITY_V4/FEW_SHOTS/EXP_BASELINES/eda/runs/EVAL_agn_LLAMA/eval.csv',
 '/home/yonatanto/work/theza/EMNLP_EXPLAINABILITY_V4/FEW_SHOTS/EXP_BASELINES/eda/runs/EVAL_imdb_LLAMA/eval.csv',
 '/home/yonatanto/work/theza/EMNLP_EXPLAINABILITY_V4/FEW_SHOTS/EXP_BASELINES/eda/runs/EVAL_agn_MISTRAL/eval.csv',
 '/home/yonatanto/work/theza/EMNLP_EXPLAINABILITY_V4/FEW_SHOTS/EXP_BASELINES/eda/runs/EVAL_emotions_LLAMA/eval.csv',
 '/home/yonatanto/work/theza/EMNLP_EXPLAINABILITY_V4/FEW_SHOTS/EXP_BASELINES/eda/runs/EVAL_rtn_MISTRAL/eval.csv',
 '/home/yonatanto/work/theza/EMNLP_EXPLAINABILITY_V4/FEW_SHOTS/EXP_BASELINES/eda/runs/EVAL_rtn_LLAMA/eval.csv',
 '/home/yonatanto/work/theza/EMNLP_EXPLAINABILITY_V4/FEW_SHOTS/EXP_BASELINES/eda/runs/EVAL_emotions_MISTRAL/eval.csv'
]

for  f in urls:
    data = pd.read_csv(f)

    p = ast.literal_eval(data.pred.item())
    gt = ast.literal_eval(data.labels.item())

    p1  = p
    if "2" not in data.labels.item():
        p1 = ['1' if i == 'P' else '0' for i in p]
    def calculate_match_percentage(list1, list2):
        if len(list1) != len(list2):
            raise ValueError("Both lists must have the same length.")

        total_items = len(list1)
        matches = sum(1 for i in range(total_items) if list1[i] == list2[i])

        return (matches / total_items) * 100
    print(f)
    print(calculate_match_percentage(p1, gt))