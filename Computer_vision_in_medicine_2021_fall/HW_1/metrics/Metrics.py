from sklearn.metrics import recall_score, f1_score, precision_score, jaccard_score, roc_curve, auc
import numpy as np
import tensorflow as tf

def get_Recall(y_truth, y_pred):
  y_pred = y_pred >0.5
  y_pred = y_pred.astype(int)
  by_class = recall_score(y_truth, y_pred, average=None)
  mean = by_class.mean()
  weighted = recall_score(y_truth, y_pred, average='weighted')
  return {"metric": "Recall", "by class": np.round(by_class,3).tolist(), "mean": np.round(mean,3), "weighted": np.round(weighted,3)}

def get_Precision(y_truth, y_pred):
  y_pred = y_pred >0.5
  y_pred = y_pred.astype(int)
  by_class = precision_score(y_truth, y_pred, average=None)
  mean = by_class.mean()
  weighted = precision_score(y_truth, y_pred, average='weighted')
  return {"metric": "Precision", "by class": np.round(by_class,3).tolist(), "mean": np.round(mean,3), "weighted": np.round(weighted,3)}

def get_F1(y_truth, y_pred):
  y_pred = y_pred >0.5
  y_pred = y_pred.astype(int)
  by_class = f1_score(y_truth, y_pred, average=None)
  mean = by_class.mean()
  weighted = f1_score(y_truth, y_pred, average='weighted')
  return {"metric": "F1", "by class": np.round(by_class,3).tolist(), "mean": np.round(mean,3), "weighted": np.round(weighted,3)}

def get_AUC(y_truth, y_pred):
  m = tf.keras.metrics.AUC()
  m.update_state(y_truth, y_pred)
  return m.result().numpy()

def get_mIoU(y_truth, y_pred):
  m = tf.keras.metrics.MeanIoU(num_classes=2)
  m.update_state(y_truth, y_pred)
  return m.result().numpy()
