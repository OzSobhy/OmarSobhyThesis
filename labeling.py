import pandas as pd

def add_label_to_csv(input_file_path, label_file_path, output_file_path):
    # Read the first CSV file (without label)
    input_df = pd.read_csv(input_file_path)
    
    # Read the second CSV file (with label)
    label_df = pd.read_csv(label_file_path)
    
    # Merge the dataframes on the timestamp column
    merged_df = pd.merge(input_df, label_df[['timestamp', 'label']], on='timestamp', how='left')
    
    # Write the merged dataframe to the output CSV
    merged_df.to_csv(output_file_path, index=False)
    print(f"Label added successfully. Output saved to {output_file_path}")

# Example usage:

input_file= r'C:\Users\osobh\GUC\Bachelor Thesis\lstm results\lstm_final_results.csv' # Replace with the path to the first CSV file
label_file= r'C:\Users\osobh\GUC\Bachelor Thesis\NAB\results\numenta\realAWSCloudwatch\numenta_ec2_cpu_utilization_24ae8d.csv'# Replace with the path to the second CSV file
output_file = r'C:\Users\osobh\GUC\Bachelor Thesis\lstm results\lstmLabel.csv'  # Replace with the path to save the output file

add_label_to_csv(input_file, label_file, output_file)
