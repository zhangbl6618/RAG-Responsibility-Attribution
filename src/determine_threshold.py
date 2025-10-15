import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def dynamic_threshold(y_scores_2D, y_true_2D):
    """
    Apply dynamic thresholding using K-means clustering.
    
    Args:
        y_scores_2D: List of lists containing responsibility scores
        y_true_2D: List of lists containing true labels
        
    Returns:
        Various evaluation metrics and predicted labels
    """
    tn_list, fp_list, fn_list, tp_list = [], [], [], []
    y_pred, y_true = [], []
    
    for idx in range(len(y_scores_2D)):
        iter_scores = y_scores_2D[idx]
        iter_labels = y_true_2D[idx]
        
        # Use K-means to find a natural threshold by clustering scores into 2 groups
        kmeans = KMeans(n_clusters=2, random_state=42)
        probability_array = np.array(iter_scores).reshape(-1, 1)
        kmeans.fit(probability_array)
        
        # The cluster with the higher center is considered the "positive" class
        centers = kmeans.cluster_centers_.flatten()
        label_for_larger_center = np.argmax(centers)
        
        # Assign binary predictions based on cluster membership
        y_pred_sub = np.zeros_like(kmeans.labels_)
        y_pred_sub[kmeans.labels_ == label_for_larger_center] = 1
        
        # Collect predictions and true labels
        y_pred.extend(y_pred_sub.tolist())
        y_true.extend(iter_labels)
        
        # Calculate confusion matrix metrics for this item
        tn, fp, fn, tp = confusion_matrix(iter_labels, y_pred_sub.tolist()).ravel()
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        tp_list.append(tp)

    # Calculate overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)  # False positive rate
    fnr = fn / (fn + tp)  # False negative rate

    return tn_list, fp_list, fn_list, tp_list, fpr, fnr, accuracy, y_pred


def evaluate_results(y_true, trace_scores_dict, variant):
    """
    Evaluate the performance of the responsibility attribution method.
    
    Args:
        y_true: List of lists containing true labels
        trace_scores_dict: Dictionary of trace scores
        variant: Which trace score variant to evaluate
        
    Returns:
        Evaluation metrics
    """
    
    # Apply dynamic thresholding and evaluate results
    tn_list, fp_list, fn_list, tp_list, fpr, fnr, accuracy, y_pred = dynamic_threshold(
        trace_scores_dict[f"variant_{variant}"], y_true
    )
    
    return tn_list, fp_list, fn_list, tp_list, fpr, fnr, accuracy, y_pred
