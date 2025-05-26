import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

class SimpleAnomalyDetector:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def is_anomaly(self, score):
        return 1 if score >= self.threshold else 0

def evaluate(file_path, threshold=0.5):
    m = pd.read_csv(file_path)

    detector = SimpleAnomalyDetector(threshold=threshold)
    predictions = []

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # Loop through the data and process sequences of labels
    i = 0
    while i < len(m):
        if m['label'][i] == 1:  # We are in an anomaly window
            # Check if there is any anomaly_score == 1.0 in this window
            anomaly_window_end = i
            while anomaly_window_end < len(m) and m['label'][anomaly_window_end] == 1:
                anomaly_window_end += 1
            
            # Check for anomaly score of 1.0 within the window
            if any(m['anomaly_score'][i:anomaly_window_end] > 0.5):
                true_positives += 1
                true_negatives += (anomaly_window_end - i - 1)  # The rest are true negatives
            else:
                false_negatives += 1
                true_negatives += (anomaly_window_end - i - 1)  # All are false negatives in the window
            
            # Now set predictions for this window
            predictions.extend([1] * (anomaly_window_end - i))
            i = anomaly_window_end  # Move i to the end of the current window

        else:  # No anomaly, normal data point
            # Determine TN or FP for this point
            prediction = detector.is_anomaly(m['anomaly_score'][i])
            if prediction == 1:
                false_positives += 1
            else:
                true_negatives += 1

            predictions.append(prediction)
            i += 1

    # Print the results for true positives, false positives, true negatives, and false negatives
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")

    # Calculate and print precision, recall, and F1 score
    try:
        precision = precision_score(m['label'], predictions)
        recall = recall_score(m['label'], predictions)
        f1 = f1_score(m['label'], predictions)
    except ValueError:
        precision, recall, f1 = 0.0, 0.0, 0.0  # Handle cases where no positive predictions are made

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


# Example usage
evaluate(r'C:\Users\osobh\GUC\Bachelor Thesis\NAB\results\numentaTM\realAWSCloudwatch\numentaTM_rds_cpu_utilization_e47b3b.csv', threshold=0.5)

#C:\Users\osobh\GUC\Bachelor Thesis\lstm results\lstmLabel.csv