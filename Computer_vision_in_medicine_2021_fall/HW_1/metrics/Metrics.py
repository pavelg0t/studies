from sklearn.metrics import recall_score, f1_score, precision_score, jaccard_score, roc_curve, auc
import numpy as np
import tensorflow as tf

def get_Recall(y_truth, y_pred):
  y_pred = y_pred >0.5
  y_pred = y_pred.astype(int)
  by_class = np.round(recall_score(y_truth, y_pred, average=None),3)
  mean = by_class.mean()
  weighted = recall_score(y_truth, y_pred, average='weighted')
  res = { 'class ' + str(i): by_class[i] for i in range(by_class.size)} 
  res.update({"metric": "Recall", "mean": np.round(mean,3), "weighted": np.round(weighted,3)})
  return res

def get_Precision(y_truth, y_pred):
  y_pred = y_pred >0.5
  y_pred = y_pred.astype(int)
  by_class = np.round(precision_score(y_truth, y_pred, average=None),3)
  mean = by_class.mean()
  weighted = precision_score(y_truth, y_pred, average='weighted')
  res = { 'class ' + str(i): by_class[i] for i in range(by_class.size)} 
  res.update({"metric": "Precision", "mean": np.round(mean,3), "weighted": np.round(weighted,3)})
  return res

def get_F1(y_truth, y_pred):
  y_pred = y_pred >0.5
  y_pred = y_pred.astype(int)
  by_class = np.round(f1_score(y_truth, y_pred, average=None),3)
  mean = by_class.mean()
  weighted = f1_score(y_truth, y_pred, average='weighted')
  res = { 'class ' + str(i): by_class[i] for i in range(by_class.size)} 
  res.update({"metric": "F1", "mean": np.round(mean,3), "weighted": np.round(weighted,3)})
  return res

def get_AUC(y_truth, y_pred):
  m = tf.keras.metrics.AUC()
  m.update_state(y_truth, y_pred)
  return m.result().numpy()

def get_mIoU(y_truth, y_pred):
  m = tf.keras.metrics.MeanIoU(num_classes=2)
  m.update_state(y_truth, y_pred)
  return m.result().numpy()
