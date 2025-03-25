import pandas as pd

# Load the datasets
dos_df = pd.read_csv(r"C:\Users\Najihah Azman\PycharmProjects\pythonProject6\data\DoS\Flooding_Based_DDoS_Multi-class_Dataset.csv")  # Example: DoS attack dataset
mitm_df = pd.read_csv(r"C:\Users\Najihah Azman\PycharmProjects\pythonProject6\data\MITM dataset\combined_mitm_data_real_experiments.csv")  # Example: MitM attack dataset


# Rename label columns for consistency
dos_df = dos_df.rename(columns={"Label": "Attack_Label"})
mitm_df = mitm_df.rename(columns={
    "Label: 1(Normal), 1000 (MITM attacking controller), 2000 (MITM attacking router), 500 (MITM  two-way attack)": "Attack_Label"
})

# Check the datasets
print("DoS Dataset Columns:", dos_df.columns)
print("MitM Dataset Columns:", mitm_df.columns)

# Define label mapping for DoS dataset based on the actual labels
dos_label_mapping = {
    "Normal": 0,            # Normal traffic
    "SYN Flooding": 1,      # SYN Flooding attack
    "ACK Flooding": 2,      # ACK Flooding attack
    "HTTP Flooding": 3,     # HTTP Flooding attack
    "UDP Flooding": 4       # UDP Flooding attack
}

# Apply mapping (Convert DoS labels to numbers)
dos_df["Attack_Label"] = dos_df["Attack_Label"].map(dos_label_mapping)

# Define label mapping for MitM dataset
mitm_label_mapping = {
    1: 0,      # Normal traffic (same as DoS "Normal")
    1000: 5,   # MITM attacking controller
    2000: 6,   # MITM attacking router
    500: 7     # MITM two-way attack
}

# Apply mapping to MitM dataset
mitm_df["Attack_Label"] = mitm_df["Attack_Label"].replace(mitm_label_mapping)
# Add missing columns in both datasets to match each other
for col in dos_df.columns:
    if col not in mitm_df.columns:
        mitm_df[col] = 0  # Fill missing features in MitM dataset with 0

for col in mitm_df.columns:
    if col not in dos_df.columns:
        dos_df[col] = 0  # Fill missing features in DoS dataset with 0

# Ensure both datasets have the same column order before merging
dos_df = dos_df[sorted(dos_df.columns)]
mitm_df = mitm_df[sorted(mitm_df.columns)]

# Combine both datasets
combined_df = pd.concat([dos_df, mitm_df], ignore_index=True)

# Move 'Attack_Label' to the last column
cols = [col for col in combined_df.columns if col != "Attack_Label"] + ["Attack_Label"]
combined_df = combined_df[cols]



# Save the final dataset
combined_df.to_csv("combined_dataset.csv", index=False)

# Check results
print("Combined Dataset Shape:", combined_df.shape)
print(combined_df["Attack_Label"].value_counts())  # Check label distribution


# Step 3: Assign Multi-Class Labels
# Create a new column 'Combined_Attack_Label' with default value "No attack"
combined_df['Combined_Attack_Label'] = 'No attack'

# Update to "Both DoS & MitM attacks" where both DoS and MitM labels are present
combined_df.loc[(combined_df['Attack_Label'] != 0), 'Combined_Attack_Label'] = 'Both DoS & MitM attacks'

# Update to "Only DoS attacks" where only DoS attacks are present
combined_df.loc[(combined_df['Attack_Label'] != 0) & (combined_df['Attack_Label'] < 5), 'Combined_Attack_Label'] = 'Only DoS attacks'

# Update to "Only MitM attacks" where only MitM attacks are present
combined_df.loc[(combined_df['Attack_Label'] >= 5), 'Combined_Attack_Label'] = 'Only MitM attacks'

# Update to "No attack" where the label is "Normal" (0) in both DoS and MitM
combined_df.loc[combined_df['Attack_Label'] == 0, 'Combined_Attack_Label'] = 'No attack'

# Step 4: Move 'Combined_Attack_Label' to the last column
cols = [col for col in combined_df.columns if col != "Combined_Attack_Label"] + ["Combined_Attack_Label"]
combined_df = combined_df[cols]

# Step 5: Save the final dataset to CSV
combined_df.to_csv("combined_dataset_with_multi-classes.csv", index=False)

# Check the results
print("Combined Dataset Shape:", combined_df.shape)
print(combined_df["Combined_Attack_Label"].value_counts())  # Check label distribution