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
print(f"Summary saved to {output_path}")
