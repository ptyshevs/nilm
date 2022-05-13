import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from functools import partial


def calculate_metrics(groundtruth, round=3, **predictions):
    metric2fn = {'r2': r2_score, 'MAE': mean_absolute_error, 'RMSE': partial(mean_squared_error, squared=False)}
    
    metrics_dict = {}
    for metric, fn in metric2fn.items():
        out = []
        for y_pred in list(predictions.values()):
            out.append(fn(groundtruth, y_pred))
        metrics_dict[metric] = out

    results = pd.DataFrame(metrics_dict, index=list(predictions.keys())).T

    if round is not None:
        return results.round(round)
    return results
