import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
died=0
alive=0
data = pd.read_csv('covid.csv',delimiter=',')
data=data.replace(2,0)
data=data.replace(1,1)
data1=data[['diabetes','age','copd','pneumonia','asthma','other_disease','cardiovascular','obesity']]
predict="date_died"
x=np.array(data1)
dead=np.array(data[predict],dtype=object)
for i in range(len(dead)):
    if dead[i]=='9999-99-99':
        dead[i]='0'
        alive=alive+1
    else:
        dead[i]='1'
        died=died+1


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, dead, test_size=0.2)

model = LogisticRegression(solver='liblinear', C=50, random_state=0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)

print(acc)

cm = confusion_matrix(y_test, model.predict(x_test))

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted alive', 'Predicted dead'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual alive', 'Actual dead'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

report = classification_report(y_test, y_pred)
print('report:', report, sep='\n')

ad=d=c=0

for i in range(len(y_pred)):
    if y_test[i]=='1':
        d=d+1
    if y_pred[i]=='1':
        ad=ad+1
    if y_pred[i]=='1' and y_test[i]=='1' :
        c=c+1

print("Predicted Dead: ", ad)
print("Originally Dead:(as per data) ", d)
print("Correctly Predicted Dead: ", c)