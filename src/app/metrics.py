"""
Metrics

******************* accuracy ***************************
accuracy : 0.9999238559354299
******************* roc ***************************
roc : 0.999897575964493
******************* f1_score ***************************
f1_score : 0.9998975654727353
******************* f1_micro ***************************
f1_micro : 0.9999238559354299
******************* f1_macro ***************************
f1_macro : 0.9999184864385753

"""
import sklearn.metrics as metrics

model_metrics = {
    "accuracy": metrics.accuracy_score,
    "roc":metrics.roc_auc_score,
    "f1_score":metrics.f1_score,
    "f1_micro": lambda y1,y2: metrics.f1_score(y1,y2,average = "micro"),
    "f1_macro" :lambda y1,y2: metrics.f1_score(y1,y2,average = "macro")
    }

