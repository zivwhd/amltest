import pandas as pd

from config.config import MetricsMetaData
from config.types_enums import ModelBackboneTypes, EvalMetric, DirectionTypes

data = pd.read_csv("test.csv")
new_data = []
for task in list(data.task.unique()):
    task_data = data[data["task"] == task]
    for model in [ModelBackboneTypes.ROBERTA, ModelBackboneTypes.DISTILBERT, ModelBackboneTypes.BERT]:
        for metric in EvalMetric:
            col = f"{model.value}_{metric.value}"
            eval_data = task_data[["function", col]].reset_index(drop = True)

            eval_data['Order'] = eval_data[col].rank(
                ascending = MetricsMetaData.directions[metric.value] == DirectionTypes.MIN.value, method = 'min')

            for index, row in eval_data.iterrows():
                new_data.append(
                    dict(task = task, model = model.value, metric = metric.value, function = row["function"],
                         rank = row["Order"], n_items = len(eval_data)))

            a = 1

new_df = pd.DataFrame(new_data)

print(new_df[(new_df["function"] == "SLVX") & (new_df["rank"] < 4)])
# new_df[(new_df["function"] == "fAML") & (new_df["rank"] > 1)].describe()
#
# print(new_df[(new_df["function"] == "fAML") & (new_df["rank"] > 1)])
# print(new_df[(new_df["function"] == "pAML") & (new_df["rank"] > 2)])
#
# new_df.groupby("function")["rank"].mean()
#
new_df["mpr"] = new_df.apply(lambda x: 1 - ((x["rank"] - 1) / x["n_items"]), axis = 1)

function = ['fAML', 'pAML', 'SIG', 'ALTI', 'DCMP', 'SHAP', 'SLVX', 'LIFT', 'IG', 'GXI', 'GLOB', 'LIME']
# # %%
models_tables = new_df.groupby(['model', 'function'])["mpr"].mean().reset_index()
task_tables = new_df.groupby(['task', 'function'])["mpr"].mean().reset_index()
metric_tables = new_df.groupby(['metric', 'function'])["mpr"].mean().reset_index()


for f in function:
    print(f" & {f}")

    for m in ['ROBERTA', 'DISTILBERT', 'BERT']:
        r = models_tables[(models_tables['model'] == m) & (models_tables['function'] == f)]['mpr'].item()

        print(f" & {r:.3f}")

    for t in ['sst2', 'RTN', 'IMDB', 'AGN', 'EMR']:
        if (f == "SLVX") and (t in ['IMDB', 'AGN']):
            print("& -")
        else:
            r = task_tables[(task_tables['task'] == t) & (task_tables['function'] == f)]['mpr'].item()
            print(f" & {r:.3f}")
    for m in ['SUFFICIENCY', 'EVAL_LOG_ODDS', 'COMPREHENSIVENESS', 'AOPC_SUFFICIENCY', 'AOPC_COMPREHENSIVENESS']:
        r = metric_tables[(metric_tables['metric'] == m) & (metric_tables['function'] == f)]['mpr'].item()
        print(f" & {r:.3f}")

    print(" \\\\")
    print("")  # %

# for i in ['model', 'task', 'metric']:
#     plt.clf()
#     sns.set_style("darkgrid")
#     # Plotting
#     q = new_df.groupby([i, 'function'])["mpr"].mean().reset_index()
#     pivot_df = q.pivot(index='function', columns=i, values='mpr')
#     pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=sns.color_palette("husl"))
#     plt.xlabel('Functions', fontsize=12, fontweight='bold')
#     plt.ylabel(i, fontsize=12, fontweight='bold')
#     plt.title(f'Results per function per {i}', fontsize=14, fontweight='bold')
#
#     # Rotate x-axis labels
#     plt.xticks(rotation=45, fontsize=10)
#
#     plt.legend(title='Category', fontsize=10)
#     plt.tight_layout()  # Adjust layout to prevent labels from being cut off
#     # plt.show()
#     plt.savefig(f"{i}.png")

#
# plt.clf()
# sns.set_style("darkgrid")
# # Plotting
# q = new_df.groupby(['function'])["mpr"].mean().reset_index()
# # pivot_df = q.pivot(index='function', columns=, values='mpr')
# q.plot(kind = 'bar', stacked = True, figsize = (10, 6), color = sns.color_palette("husl"))
# plt.xlabel('Functions', fontsize = 12, fontweight = 'bold')
# plt.ylabel(i, fontsize = 12, fontweight = 'bold')
# plt.title(f'Results per function', fontsize = 14, fontweight = 'bold')
#
# # Rotate x-axis labels
# plt.xticks(rotation = 45, fontsize = 10)
#
# plt.legend(title = 'Category', fontsize = 10)
# plt.tight_layout()  # Adjust layout to prevent labels from being cut off
# # plt.show()
# plt.savefig(f"{i}.png")
#
# # --------------


q =  new_df.groupby(['function'])["mpr"].mean().reset_index()
q.to_csv("./a.csv")

a = pd.read_csv("./a.csv")
a = a.sort_values("order")
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Plotting
plt.figure(figsize = (10, 6))
sns.barplot(x = 'function', y = 'mpr', data = a, palette = "Blues_d")

# Add values on top of bars
for index, value in enumerate(a['mpr']):
    plt.text(index, value + 0.02, f"{value:.3f}", ha = 'center', fontsize = 12)

plt.xlabel('Explanation methods', fontsize = 14)
plt.ylabel('MPR', fontsize = 14)
plt.title('AVG MPR per explanation method', fontsize = 16)
plt.ylim(0, 1)  # Setting y-axis limits from 0 to 1
plt.xticks(fontsize = 12, rotation = 45)
plt.yticks(fontsize = 12)
plt.tight_layout()
plt.show()
