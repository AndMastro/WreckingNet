from sklearn.metrics import confusion_matrix
import numpy as np


cm = np.array( [[11375, 194, 36, 126, 251],
                [170, 11307   , 112   , 343    , 50],
                [105         , 0, 11830    , 42     , 5],
                [71    , 37    , 51  , 11737    , 86],
                [122   , 154    , 15   , 175 , 11516]])

recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)

meanRecall = np.mean(recall)
meanPrecision = np.mean(precision)
f1 = (2*(meanPrecision*meanRecall))/(meanPrecision+meanRecall)

print("Precision: " + str(meanPrecision))
print("Recall: " + str(meanRecall))
print("F1: " + str(f1))


#mean metrics of k-fold
print("==================")

print("K fold metrics:")

precKFold = [0.9224027130123359,#
0.9328118908167555,#
0.9175936857528437,#
0.9470498157964325,#
0.9643294204204714]#

recallKFold = [0.9215985905554185,#
0.9313303288265733,#
0.9140807474140807,#
0.9460812953843586,#
0.9641962944416624]#

f1KFold = [0.9220004764551167,#
0.9320705210720756,#
0.9158338478800647,#
0.946565307844272,#
0.9642628528362289]#


print("Precision: " + str(np.mean(precKFold)))
print("Recall: " + str(np.mean(recallKFold)))
print("F1: " + str(np.mean(f1KFold)))