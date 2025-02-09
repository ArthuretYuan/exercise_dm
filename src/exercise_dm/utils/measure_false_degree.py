# This file contains the function that measures the false degree of a classifier.

def measure_false_degree(pred_labels, true_labels):
    """
    There are 5 classes, if the predicted label is different from the true label, the error is classified as:
    slight error: 1 class difference
    moderate error: 2 class difference
    severe error: 3 class difference
    critical error: 4 class difference
    """
    correct_count = 0
    incorrect_count = 0
    false_degree = {'slight error': 0, 'moderate error': 0, 'severe error': 0, 'critical error': 0}
    for pred_label, true_label in zip(pred_labels, true_labels):
        if pred_label == true_label:
            correct_count += 1
        else:
            incorrect_count += 1
            diff = abs(pred_label - true_label)
            if diff == 1:
                false_degree['slight error'] += 1
            elif diff == 2:
                false_degree['moderate error'] += 1
            elif diff == 3:
                false_degree['severe error'] += 1
            elif diff == 4:
                false_degree['critical error'] += 1
    return correct_count, incorrect_count, false_degree