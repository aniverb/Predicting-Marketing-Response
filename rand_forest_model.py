import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

train = pd.read_csv("C:/Users/ahenry23/Documents/train_a_reduced_200.csv")
test = pd.read_csv("C:/Users/ahenry23/Documents/test_200.csv")

train_target = train['target']
test_target = test['target']

cat_cols = train.iloc[:,-26:train.shape[1]]
train = train.iloc[:,2:train.shape[1]]
train = pd.get_dummies(train, columns=cat_cols)
test = test.iloc[:,2:test.shape[1]]
test = pd.get_dummies(test, columns=cat_cols)

#Random forest classifier
rfc = RandomForestClassifier(max_features="sqrt", n_estimators=130)
rfc.fit(train, train_target)
test_pred = rfc.predict(test)
#train_pred = rfc.predict(train)     #For testing on the train set

#Measuring accuracy, from Nole's Logistic Regression code
correct = 0
for k in range(len(test_pred)):
    if test_pred[k] == test_target[k]:
        correct = correct + 1
print str(correct / float(len(test_pred)))       # Accuracy on testing set - 77.67%

'''correct = 0                                     
for k in range(len(train_pred)):
    if train_pred[k] == train_target[k]:
        correct = correct + 1
print str(correct / float(len(train_pred)))'''   # Accuracy on training set - 100%

#ROC curve
pred = rfc.predict_proba(test)[:,1]
fpr, tpr, _ = roc_curve(test_target, pred)

#pred = rfc.predict_proba(train)[:,1]
#fpr, tpr, _ = roc_curve(train_target, pred)

plt.plot(fpr, tpr);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('Random Forest ROC Curve');
plt.show()

#AUC
auc(fpr, tpr) #0.693724 for test, 1.0 for train
