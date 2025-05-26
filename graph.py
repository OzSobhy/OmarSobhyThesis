import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
def read_csv(file_path):
    m = pd.read_csv(file_path)
    m['timestamp'] = pd.to_datetime(m['timestamp'])
    return m

# Plot the graphs on the same graph
def plot_graph(m):
    plt.figure(figsize=(12, 6))

    # Plot value
    plt.plot(m['timestamp'], m['value'], label='Value', color='b')

    # Plot anomaly score * 100
    plt.plot(m['timestamp'], m['anomaly_score'] * 100, label='Anomaly Score * 100', color='g')

    # Plot label * 100
    plt.plot(m['timestamp'], m['label'] * 100, label='Label * 100', color='r')

    # Add title and labels
    plt.title('Value, Anomaly Score * 100, and Label * 100 over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Value / Anomaly Score / Label')

    # Add legend
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


# Main function to execute the code
if __name__ == "__main__":
    file_path = r'C:\Users\osobh\GUC\Bachelor Thesis\lstm results\lstmLabel.csv'  # Replace with the path to your CSV file
    m = read_csv(file_path)
    plot_graph(m)

# 
# C:\Users\osobh\GUC\Bachelor Thesis\lstm results\lstmLabel.csv