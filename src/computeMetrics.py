from sklearn.metrics import confusion_matrix
import numpy as np


cm = np.array( [[11755   , 62  ,   6  ,  13 ,  146],
 [  243 ,11569   , 42 ,   19,   109],
 [   24    , 7 ,11888  ,  63  ,   0],
 [   50 ,   10    ,74 ,11832 ,   16],
 [  153  , 112   , 11  ,  19, 11687]])

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

precKFold = [0.9611817210299931,##
0.971973124444364,##
0.9797840753208193,##
0.9736974296698258,##
0.9804086256769784]##

recallKFold = [0.9603003754693367,##
0.971653713619148,##
0.9796796796796796,##
0.9734418997284987,##
0.9803204807210817]##

f1KFold = [0.9607408461218391,##
0.9718133927861636,##
0.9797318747192718,##
0.9735696479321159,##
0.9803645512177434]##


print("Precision: " + str(np.mean(precKFold)))
print("Recall: " + str(np.mean(recallKFold)))
print("F1: " + str(np.mean(f1KFold)))