import pandas as pd
from pathlib import Path

# Define base paths
base_path = Path("OUT")
output_path = base_path / "msummary.csv"

# List all results_df.csv files recursively
csv_files = list(base_path.glob("*/*/results.csv"))

print(f"found: {csv_files}")
# Collect grouped DataFrames
all_grouped = []

for csv_file in csv_files:
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Group by the specified columns and compute mean
    grouped = (
        df.groupby(['task', 'attribution_scores_function', 'evaluation_metric', 'explained_model_backbone'])['metric_result']
        .mean()
        .reset_index()
        .rename(columns={'metric_result': 'mean_metric_result'})
    )
    
    # Add a column with the file path
    grouped['file_path'] = str(csv_file)
    
    # Append to the list
    all_grouped.append(grouped)

# Concatenate all grouped DataFrames
summary_df = pd.concat(all_grouped, ignore_index=True)

# Save to OUT/summary.csv
summary_df.to_csv(output_path, index=False)

########

all_grouped_df = summary_df.sort_values("file_path")
pgrouped = (
    all_grouped_df.groupby(["task", "attribution_scores_function", "explained_model_backbone", "evaluation_metric"])
    .last()
    .reset_index()
)

# Pivot to wide format
pivoted = pgrouped.pivot_table(
    index=["task", "attribution_scores_function", "explained_model_backbone"],
    columns="evaluation_metric",
    values="metric_result"
).reset_index()

# Rename columns
pivoted = pivoted.rename(columns={
    "SUFFICIENCY": "Shuff",
    "EVAL_LOG_ODDS": "LO",
    "COMPREHENSIVENESS": "Comp",
    "AOPC_SUFFICIENCY": "AS",
    "AOPC_COMPREHENSIVENESS": "AC"
})

# Ensure the columns order
cols_order = ["task", "attribution_scores_function", "explained_model_backbone", "Shuff", "LO", "Comp", "AS", "AC"]
pivoted = pivoted[cols_order]

# Save to psummary.csv

pivoted.to_csv(base_path / "psummary.csv", index=False)

print(f"Summary saved to {output_path}")
