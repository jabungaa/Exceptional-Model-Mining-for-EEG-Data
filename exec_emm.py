import pandas as pd
from time import time
from beam_search import EMM, as_string

start = time()

# Load and prepare data
df = pd.read_csv("sub_info_and_features.csv")

# Define descriptive features and EEG features
descriptive_features = ["Gender", "Age", "MMSE"]
eeg_features = [col for col in df.columns if col not in ["participant_id", "Gender", "Age", "Group", "MMSE"]]

# Create specific binary targets and define analysis runs
df["is_A"] = (df["Group"] == "A").astype(int)
df["is_F"] = (df["Group"] == "F").astype(int)

# List of targets to analyze in separate EMM runs
targets_to_analyze = [
    {"name": "A", "col": "is_A"},
    {"name": "F", "col": "is_F"}
]

# Loop through each target and run EMM
for target in targets_to_analyze:
    target_name = target["name"]
    target_col = target["col"]

    print(f"\n{"=" * 25}")
    print(f"  RUNNING EMM FOR GROUP {target_name}")
    print(f"{"=" * 25}")

    # Set EMM parameters and run (away from all the problems... (ha...))
    results = EMM(
        w=50,
        d=6,
        q=50,
        catch_all_description=[],
        df=df,
        features=descriptive_features,
        eeg_features=eeg_features,
        target=target_col,
        n_chunks=20
    )

    # Print results for the current run
    print(f"\n--- Top Exceptional Subgroups for Group {target_name} ---")
    if not any(results.get_values()):
        print("No exceptional subgroups found for this group with the current settings.")
        continue

    for (quality, description, _) in results.get_values():
        subgroup_df = df.query(as_string(description))
        num_patients = len(subgroup_df)
        num_in_group = subgroup_df[target_col].sum()
        group_rate = (num_in_group / num_patients) * 100

        print(
            f"Quality: {quality:.4f} | "
            f"Description: '{as_string(description)}' | "
            f"Patients: {num_patients} | "
            f"Group {target["name"].split()[0]} Rate: {group_rate:.1f}%"
        )

print(f"\nTIME {time() - start}s")
