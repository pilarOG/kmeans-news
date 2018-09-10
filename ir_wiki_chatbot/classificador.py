import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn import tree
import pickle

data = pd.read_csv('labeled_data.csv', header = 0, delimiter = ",", encoding = 'utf-8',decimal=",")

data = shuffle(data)

feat1 = data['bool_a_n']
feat2 = data['bool_v_i']
feat3 = data['emb_a_n']
feat4 = data['emb_i_r']
feat5 = data['emb_i_v']
feat6 = data['lm_score']

y = np.array(data['label'])

X = np.zeros((len(y), 6))

for x in range(0, len(y)):
    X[x,0] = float(feat1[x])
    X[x,1] = float(feat2[x])
    X[x,2] = float(feat3[x])
    X[x,3] = float(feat4[x])
    X[x,4] = float(feat5[x])
    X[x,5] = float(feat6[x])

y = np.array(data['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#clf = SGDClassifier(loss="hinge", penalty="l2")
#clf = tree.DecisionTreeClassifier()
clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(30,), random_state=1)
clf.fit(X_train, y_train)

ac = 0
for x in range(0, len(X_test)):
    y_pred = clf.predict(X_test[x])
    print y_pred, y_test[x]
    if y_pred == y_test[x]:
        ac += 1
print float(ac)*100/float(len(X_test))

pickle.dump(clf, open("ir_classification.pckl", "wb"))
