import json
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

def exportMetrics(
    y_true,
    y_pred,
    labels,
    model_name,
):
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": report[cls]["support"]
            }
            for cls in labels if cls in report
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist()
    }

    filename = f"{model_name}_metrics.json"
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics file {filename} is saved")
    return metrics
