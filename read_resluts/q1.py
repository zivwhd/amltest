aml = dict(
emotions_AOPC_COMPREHENSIVENESS="/home/yonatanto/work/theza/MY_PAPERS_CODE/EXP_AML/inference/inference_aml/PAML_emotions_ROBERTA_ROBERTA_AOPC_COMPREHENSIVENESS_1717724178_pt_1718150137.006651.pkl",
emotions_AOPC_SUFFICIENCY="/home/yonatanto/work/theza/MY_PAPERS_CODE/EXP_AML/inference/inference_aml/PAML_emotions_ROBERTA_ROBERTA_AOPC_SUFFICIENCY_1717717846_pt_1718150741.0831022.pkl",
sst_AOPC_COMPREHENSIVENESS="/home/yonatanto/work/theza/MY_PAPERS_CODE/EXP_AML/inference/inference_aml/PAML_sst_ROBERTA_ROBERTA_AOPC_COMPREHENSIVENESS_1717750418_pt_1718149842.638299.pkl",
sst_AOPC_SUFFICIENCY="/home/yonatanto/work/theza/MY_PAPERS_CODE/EXP_AML/inference/inference_aml/PAML_sst_ROBERTA_ROBERTA_AOPC_SUFFICIENCY_1717746666_pt_1718150433.8668094.pkl"
)


df = {k: pd.read_pickle(v) for k, v in aml.items()}

def print_res_emotions(i):
 print("alti - " + df["emotions"].iloc[i]["alti"])
 print("sig - " + df["emotions"].iloc[i]["sequential_integrated_gradients"])


def print_res_sst(i):
    print("alti - " + df["sst"].iloc[i]["alti"])
    print("sig - " + df["sst"].iloc[i]["sequential_integrated_gradients"])


