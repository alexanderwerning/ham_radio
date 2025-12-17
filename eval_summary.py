import json
import numpy as np
import matplotlib.pyplot as plt
from fire import Fire
from pathlib import Path

def compute_roc(tp, fp, tn, fn, thresholds):
    # sum from back to front
    fn[tp==0] = 1
    tpr = tp / (tp+fn)  # true positive rate
    # sum from front to back
    fp[tn==0] = 1
    tnr = tn / (tn+fp) # true negative rate
    fpr = 1 - tnr  # false positive rate
    order = np.argsort(fpr)  # not necessary
    roc = np.vstack([fpr[order], tpr[order]]).T
    roc_auc = np.trapz(tpr[order], fpr[order])
    eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
    eer_threshold = thresholds[eer_idx]
    eer = fpr[eer_idx]
    recall_eer = tpr[eer_idx]
    tp_eer = tp[eer_idx]
    fp_eer = fp[eer_idx]
    precision_eer = tp_eer / (tp_eer+fp_eer) if (tp_eer+fp_eer) > 0 else 0.0
    return roc, roc_auc, eer, eer_threshold, recall_eer, precision_eer

def evaluate(model_name="SADModel_12", storage_root='/net/vol/werning/ham_radio/models/ham_radio'):

    file = Path(storage_root)/model_name/'tp_fp_tn_fn_test.json'
    if not file.exists():
        file = Path(storage_root)/model_name/'tp_fp_tn_fn_eval.json'
        if not file.exists():
            raise FileNotFoundError(f"Could not find evaluation file for model {model_name}")
    results = json.load(open(file, 'r'))
    output = {}
    k, v = 'sad', results['sad']
    tp_list, fp_list, tn_list, fn_list, threshold_list = [], [], [], [], []
    out = {}
    for threshold, values in v.items():
        tp, fp, tn, fn = values
        tp_list.append(tp)
        fp_list.append(fp)
        tn_list.append(tn)
        fn_list.append(fn)
        threshold_list.append(threshold)
        recall = tp/(tp+fn) if tp > 0 else 0
        precision = tp/(tp+fp) if tp > 0 else 1
        f1 = 2*recall*precision/(recall+precision) if recall+precision > 0 else 0
        out[f'{float(threshold):0.4f}'] = {'recall': f'{recall:0.4f}', 'precision': f'{precision:0.4f}', 'f1-score': f'{f1:0.4f}'}
    output[k] = out
    json.dump(output, open(f'results_{model_name}.json', 'w'))
    thresholds = [float(o) for o in output['sad']]
    plt.plot(thresholds, [float(output['sad'][t]['recall']) for t in output['sad']])
    plt.title(f"Recall {model_name}")
    plt.savefig(f'recall_{model_name}.png')
    plt.clf()
    plt.plot(thresholds, [float(output['sad'][t]['precision']) for t in output['sad']])
    plt.title(f"Precision {model_name}")
    plt.savefig(f'precision_{model_name}.png')
    roc, roc_auc, eer, eer_threshold, recall_eer, precision_eer = compute_roc(np.asarray(tp_list), np.asarray(fp_list), np.asarray(tn_list), np.asarray(fn_list), thresholds)
    print(f'roc auc {roc_auc}, eer {eer}, eer_threshold {eer_threshold}, recall eer {recall_eer} precision_eer {precision_eer}')
    plt.clf()
    plt.plot(*roc.T)
    plt.ylim(0.9, 1)
    plt.xlim(0, 0.3)
    plt.grid(True)
    plt.title(f"ROC {model_name}")
    plt.savefig(f'roc_{model_name}.png')

if __name__ == "__main__":
    Fire(evaluate)