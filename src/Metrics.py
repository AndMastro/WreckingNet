import numpy as np

def compute_metrics(matrix, class_id):
    tp = matrix[class_id, class_id]
    fp = matrix[:,class_id].sum() - tp # column not class
    fn = matrix[class_id,:].sum() - tp # row not class
    tn = matrix.sum() - (tp + fn +  fp)

    return tp, tn, fp, fn

def accuracy(confusion_matrix, class_id):
    tp, tn, fp, fn = compute_metrics(confusion_matrix, class_id)
    return (tp+tn)/(tp + tn + fp + fn)

def recall(confusion_matrix, class_id):
    tp, tn, fp, fn = compute_metrics(confusion_matrix, class_id)
    return tp/(tp + fn)

def precision(confusion_matrix, class_id):
    tp, tn, fp, fn = compute_metrics(confusion_matrix, class_id)
    return tp/(tp + fp)

def F1_score(confusion_matrix, class_id):
    tp, tn, fp, fn = compute_metrics(confusion_matrix, class_id)
    return (2*tp)/(2*tp + fp +fn)

#if __name__ == "__main__":
f0 = np.array([
	[11555,   136,    38,    87,   169],
	[  751, 10801,    40,    75,   318],
	[   62,    37, 11819,    51,    16],
	[   42,     3,    50, 11886,     4],
	[  346,   104,    15,    35, 11485]]
)
f1 = np.array([
	[11177,   250,   227,    88,   240],
	[   28, 11811,    97,    15,    31],
	[    3,    64, 11871,    24,    20],
	[  103,    28,   244, 11536,    71],
	[   94,   101,     5,     1, 11781]])
f2 = np.array([
	[11884,    36,     2,    14,    52],
	[  191, 11454,    30,     9,   304],
	[   52,    51, 11821,    57,     7],
	[   57,    22,    75, 11818,    16],
	[   88,   114,    13,    28, 11745]])
f3 = np.array([
	[11888,    24,     8,    43,    18],
	[  119, 11712,    68,     6,    76],
	[   83,    26, 11617,   176,    79],
	[  255,    26,   428, 11251,    21],
	[   26,    28,     6,     5, 11916]]
)
f4 = np.array([
	[11755,    62,     6,    13,   146],
	[  243, 11569,    42,    19,   109],
	[   24,     7, 11888,    63,     0],
	[   50,    10,    74, 11832,    16],
	[  153,   112,    11,    19, 11687]])

cms = [f0, f1, f2, f3, f4]
classes = list(range(5))

mapping = {
'0':"RETROIA",
'1':"COMPACTOR CASUALE",
'2':"BETONIERA",
'3':"SCAVATORE GATTO",
'4':"SCAVATORE INDIANO",
}

i = 0

for c in classes:
	key = str(c)
	print("Class:", c, mapping[key])
	acc = []
	prec = []
	rec = []
	F1 = []
	for i in range(len(cms)):
		cm = cms[i]

		acc.append(accuracy(cm, c))
		prec.append(precision(cm, c))
		rec.append(recall(cm, c))
		F1.append(F1_score(cm, c))

	acc = np.array(acc)
	prec = np.array(prec)
	rec = np.array(rec)
	F1 = np.array(F1)
	
	print('\t', "Accuracy:", acc.mean(),
			"Precision", prec.mean(),
			"Recall", rec.mean(),
			"F1", F1.mean())

	
"""
for i in range(len(cms)):
	print("Fold", i)
	cm = cms[i]
	for c in classes:
		print('\t',"Class", c, mapping[str(c)])
		acc = accuracy(cm, c)
		prec = precision(cm, c)
		rec = recall(cm, c)
		F1 = F1_score(cm, c)		
		print("Accuracy:", acc,
			"Precision", prec,
			"Recall", rec,
			"F1", F1,)


	tp, tn, fp, fn = compute_metrics(matrix, class_id)

	print("tp =", tp)
	print("fp =", fp)
	print("tn =", tn)
	print("fn =", fn)

	print("Accuracy = ", accuracy(matrix, class_id))
	print("Recall =", recall(matrix, class_id))
	print("Precision = ", precision(matrix, class_id))
	print("F1 =", F1_score(matrix, class_id))
"""
