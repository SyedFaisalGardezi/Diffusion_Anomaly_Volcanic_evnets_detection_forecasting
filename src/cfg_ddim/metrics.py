# import numpy as np
# from math import ceil
# from sklearn.metrics import f1_score, auc as _auc
# 
# def _get_segments(labels):
#     """
#     Find all contiguous runs where labels == 1.
#     Returns a list of (start, end) tuples, with `end` exclusive.
#     """
#     segs = []
#     in_seg = False
#     for i, v in enumerate(labels):
#         if v and not in_seg:
#             start = i
#             in_seg = True
#         elif not v and in_seg:
#             segs.append((start, i))
#             in_seg = False
#     if in_seg:
#         segs.append((start, len(labels)))
#     return segs
# 
# 
# def _adjust_preds_K(pred, segments, K):
#     """
#     Given a binary `pred` array (0/1) and the true‐anomaly segments,
#     apply the PA@K rule: a segment is marked positive iff at least
#     K% of its points were predicted positive. Otherwise the entire
#     segment is marked negative.
#     """
#     pred_adj = pred.copy()
#     for (s, e) in segments:
#         length = e - s
#         required = ceil(K / 100 * length)
#         if pred[s:e].sum() >= required:
#             # At least K% of that block’s points were flagged → mark whole block as 1
#             pred_adj[s:e] = 1
#         else:
#             # Didn’t reach K% → mark whole block as 0
#             pred_adj[s:e] = 0
#     return pred_adj
# 
# 
# def compute_f1_k_auc(
#     scores: np.ndarray,
#     labels: np.ndarray,
#     threshold: float,
#     k_values: np.ndarray = None
# ):
#     """
#     Compute F1_K-AUC at a fixed threshold δ.
# 
#     Args:
#         scores:     1D array of anomaly scores, length T
#         labels:     1D binary ground‐truth (0 or 1), length T
#         threshold:  scalar δ to binarize `scores`: pred0 = (scores > δ)
#         k_values:   optional 1D array of K% values in [0,100].
#                     Defaults to np.arange(0,101).
# 
#     Returns:
#         f1k_auc: area under the F1 vs K(%) curve (normalized by 100)
#         f1s:     1D array of F1 scores for each K
#         ks:      the K values used (0 through 100 by default)
#     """
#     if k_values is None:
#         ks = np.arange(0, 101)
#     else:
#         ks = np.asarray(k_values)
# 
#     # 1) Binarize once at δ
#     pred0 = (scores > threshold).astype(int)
#     segments = _get_segments(labels)
# 
#     # 2) For each K, apply PA%K → compute F1
#     f1s = []
#     for K in ks:
#         pred_k = _adjust_preds_K(pred0, segments, K)
#         f1 = f1_score(labels, pred_k)
#         f1s.append(f1)
#     f1s = np.array(f1s)
# 
#     # 3) AUC under the F1 vs K curve; since K∈[0,100], divide by 100
#     f1k_auc = _auc(ks, f1s) / 100.0
# 
#     return f1k_auc, f1s, ks
# 
# 
# def compute_roc_k_auc(
#     scores: np.ndarray,
#     labels: np.ndarray,
#     thresholds: np.ndarray = None,
#     k_values: np.ndarray = None
# ):
#     """
#     Compute ROC_K-AUC by:
#       (1) For each fixed K ∈ {0,…,100}, computing a ROC curve over δ;
#       (2) Taking the AUC of that curve (call it AUC_K);
#       (3) Averaging AUC_K over all K.
# 
#     Args:
#         scores:     1D anomaly scores, length T
#         labels:     1D binary ground‐truth (0 or 1), length T
#         thresholds: Optional 1D array of δ values to test. If None,
#                     defaults to 50 evenly-spaced points in [0, s_max],
#                     where s_max = max(scores). This exactly mirrors
#                     the paper’s “50 evenly-spaced δ” protocol.
#         k_values:   Optional 1D array of K% values in [0,100].
#                     Defaults to np.arange(0,101).
# 
#     Returns:
#         roc_k_auc: float, the mean ROC-AUC over all K (i.e. 1/101 Σ AUC_K)
#         fpr_dict:  dict mapping each K → 1D array of FPRs (sorted ascending)
#         tpr_dict:  dict mapping each K → 1D array of TPRs (sorted ascending)
#     """
#     # 1) If thresholds not provided, build 50 evenly-spaced values in [0, s_max]
#     if thresholds is None:
#         s_max = scores.max()
#         # 50 points: 0, 1/49*s_max, 2/49*s_max, …, s_max
#         thresholds = np.linspace(0.0, s_max, 50)
#     else:
#         thresholds = np.asarray(thresholds)
# 
#     # 2) If K values not provided, use {0,1,…,100}
#     if k_values is None:
#         ks = np.arange(0, 101)
#     else:
#         ks = np.asarray(k_values)
# 
#     segments = _get_segments(labels)
#     N = len(labels)
#     P = labels.sum()        # Number of true‐anomaly points
#     Nn = N - P              # Number of true‐normal points
# 
#     # Containers for each K’s ROC curve
#     fpr_dict = {}
#     tpr_dict = {}
#     aucs_per_k = []
# 
#     # 3) Loop over each K, build its own ROC curve by sweeping δ
#     for K in ks:
#         all_fpr = []
#         all_tpr = []
# 
#         # 3a) For each δ, binarize → adjust by K → compute (TPR,FPR)
#         for δ in thresholds:
#             pred0 = (scores > δ).astype(int)
#             pred_k = _adjust_preds_K(pred0, segments, K)
# 
#             tp = int(((pred_k == 1) & (labels == 1)).sum())
#             fp = int(((pred_k == 1) & (labels == 0)).sum())
# 
#             tpr = tp / P if P > 0 else 0.0
#             fpr = fp / Nn if Nn > 0 else 0.0
# 
#             all_tpr.append(tpr)
#             all_fpr.append(fpr)
# 
#         # 3b) Sort by increasing FPR so that AUC integration is valid
#         order = np.argsort(all_fpr)
#         fprs_sorted = np.array(all_fpr)[order]
#         tprs_sorted = np.array(all_tpr)[order]
# 
#         fpr_dict[K] = fprs_sorted
#         tpr_dict[K] = tprs_sorted
# 
#         # 3c) Compute AUC for this fixed K
#         auc_k = _auc(fprs_sorted, tprs_sorted)
#         aucs_per_k.append(auc_k)
# 
#     # 4) Average AUC_K over all K to get ROC_K-AUC
#     roc_k_auc = float(np.mean(aucs_per_k))
# 
#     return roc_k_auc, fpr_dict, tpr_dict
# 


# metrics.py
import numpy as np
from sklearn.metrics import f1_score, auc, confusion_matrix
from tadpak import pak

def get_fp_tp_rate(predict, actual):
    tn, fp, fn, tp = confusion_matrix(actual, predict, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return fpr, tpr

def pak_protocol(scores, labels, threshold, max_k=100):
    """
    Returns:
      area_under_f1, max_f1_at_some_k, k_of_max_f1,
      preds_for_max_f1,  fpr_list_per_k, tpr_list_per_k
    """
    f1s, fprs, tprs, preds = [], [], [], []
    ks = list(range(0, max_k + 1))

    for k in ks:
        pred_adj = pak.pak(scores, labels, threshold, k=k)
        f1s.append(f1_score(labels, pred_adj))
        fpr, tpr = get_fp_tp_rate(pred_adj, labels)
        fprs.append(fpr)
        tprs.append(tpr)
        preds.append(pred_adj)

    # area under the F1 vs (k/100) curve
    area_under_f1 = auc([k/100 for k in ks], f1s)

    max_f1 = max(f1s)
    k_max  = f1s.index(max_f1)
    preds_for_max = preds[k_max]

    return area_under_f1, max_f1, k_max, preds_for_max, fprs, tprs

def evaluate(scores, labels, validation_thresh=None):
    """
    If validation_thresh is None:
      returns (metrics_dict, flat_fpr, flat_tpr)
    else:
      returns metrics_dict only, computed at that fixed threshold.
    metrics_dict always has keys:
      - 'f1'       = area under F1(k) curve (F1_K-AUC) at best k
      - 'ROC/AUC'  = area under the big ROC cloud (all thresholds × k)
      - 'threshold' or 'thresh_max'
      - 'f1_max'   = best single-K F1 on that threshold
      - 'k'        = the K that gave that best single‐K F1
      - 'preds'    = the binary preds at that (threshold, k)
    """
    false_pos_rates = []
    true_pos_rates  = []
    f1s             = []
    max_f1s_k       = []
    preds_list      = []
    thresholds      = np.linspace(0, scores.max(), 50)
    ks_pairs        = []

    # for each threshold, run pak_protocol
    for thresh in thresholds:
        a_f1, a_f1_max, a_k_max, a_preds, fprs, tprs = pak_protocol(
            scores, labels, thresh, max_k=100
        )
        f1s.append(a_f1)
        max_f1s_k.append(a_f1_max)
        preds_list.append(a_preds)
        false_pos_rates.append(fprs)
        true_pos_rates.append(tprs)
        ks_pairs.extend([(thresh, k) for k in range(101)])

    false_pos_rates = np.array(false_pos_rates)  # shape (50,101)
    true_pos_rates  = np.array(true_pos_rates)

    # if user wants fixed validation_thresh → skip best‐threshold search
    if validation_thresh is not None:
        a_f1, a_f1_max, a_k_max, a_preds, _, _ = pak_protocol(
            scores, labels, validation_thresh, max_k=100
        )
        best = {
            'f1':         a_f1,
            'ROC/AUC':    auc(
                             false_pos_rates.flatten(),
                             true_pos_rates.flatten()),
            'f1_max':     a_f1_max,
            'preds':      a_preds,
            'k':          a_k_max,
            'thresh_max': validation_thresh
        }
        return best

    # otherwise, pick best‐threshold
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    best_f1     = f1s[best_idx]
    max_possible_f1 = max(max_f1s_k)
    k_of_max = max_f1s_k.index(max_possible_f1)

    # Build roc_max (for that best‐k curve)
    roc_max = auc(
        false_pos_rates[:, k_of_max],
        true_pos_rates[:, k_of_max]
    )

    # Flatten and sort the entire ROC cloud
    fpr_flat = false_pos_rates.flatten()
    tpr_flat = true_pos_rates.flatten()
    order = np.argsort(fpr_flat)
    fpr_flat, tpr_flat = fpr_flat[order], tpr_flat[order]

    metrics = {
        'f1':         best_f1,
        'ROC/AUC':    auc(fpr_flat, tpr_flat),
        'threshold':  best_thresh,
        'f1_max':     max_possible_f1,
        'k':          k_of_max,
        'thresh_max': best_thresh,
        'roc_max':    roc_max,
        'preds':      preds_list[best_idx]
    }
    return metrics, fpr_flat, tpr_flat

