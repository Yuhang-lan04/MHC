import numpy as np  # 导入numpy库，用于数值计算

def do_metric(y_prob, label):
    y_prob = y_prob.cpu().detach().numpy()
    average_precision_score = compute_average_precision_score(y_prob, label)
    return average_precision_score

def compute_average_precision_score(y_prob, label):

    ap_sum = np.sum(y_prob == label) / y_prob.size
    return ap_sum