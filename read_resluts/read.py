from glob import glob

import pandas as pd

from config.config import MetricsMetaData
from config.types_enums import DirectionTypes


def read_file(f):
    file_df = pd.read_csv(f)
    file_df["folder"] = "_".join(f.split("/")[:-1])
    return file_df


def read_baselines():
    baselines_files = "/home/yonatanto/work/theza/EXPLAINABILITY/EXP_BASELINES/*/runs/baselines/OUT/*/*/results.csv"
    baselines_df = pd.concat([read_file(f) for f in glob(baselines_files)])
    return baselines_df


def read_rpi():
    rpi_files = "/home/yonatanto/work/theza/EXPLAINABILITY/EXP_RPI/*/runs/OUT/*/*/results.csv"
    rpi_df = pd.concat([read_file(f) for f in glob(rpi_files)])
    rpi_df["attr_score_unction"] = "RPI"
    return rpi_df


def read_aml():
    aml_files = "/home/yonatanto/work/theza/EXPLAINABILITY/EXP_AML/*/runs/OUT/FINE_TUNE/*/*/results.csv"
    aml_df = pd.concat([read_file(f) for f in glob(aml_files)])
    aml_df["attr_score_unction"] = "AML"
    return aml_df


def read_paml():
    paml_files = "/home/yonatanto/work/theza/EXPLAINABILITY/EXP_AML/*/runs/OUT/FINE_TUNE/RESULTS_DF/*/support_results_df.csv"
    paml_df = pd.concat([read_file(f) for f in glob(paml_files)])
    paml_df = paml_df[(paml_df['epoch'] == 0) & (paml_df['step'] == 0)]
    paml_df["attr_score_unction"] = "pAML"
    return paml_df

aml = read_aml()
all_df = pd.concat([read_baselines(), read_rpi(), read_aml(), read_paml()])  #
all_df["task"].replace({"rtm": "rtn"}, inplace = True)
x = all_df.groupby(
    ['task', 'folder', "explained_model_backbone", 'eval_tokens', 'eval_metric', 'attr_score_unction']).aggregate(
    {"metric_result": ["mean", "count"]}).reset_index()

dfs = {}
for explained_model_backbone in list(x.explained_model_backbone.unique()):
    for task in list(x.task.unique()):
        for eval_metric in list(x.evaluation_metric.unique()):
            for eval_tokens in list(x.token_evaluation_option.unique()):
                dfs[f"{task}_{explained_model_backbone}_{eval_tokens}_{eval_metric}"] = x[
                    (x["task"] == task) & (x["explained_model_backbone"] == explained_model_backbone) & (
                                x["eval_tokens"] == eval_tokens) & (x["eval_metric"] == eval_metric)].sort_values(
                    by = ('metric_result', 'mean'), ascending = MetricsMetaData.directions[eval_metric] == DirectionTypes.MIN.value)
                dfs[f"{task}_{explained_model_backbone}_{eval_tokens}_{eval_metric}"].to_csv(f"./results/{task}_{explained_model_backbone}_{eval_tokens}_{eval_metric}.csv")