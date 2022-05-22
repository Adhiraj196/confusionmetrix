#true+ + true- / true+ + true- + false+ + false- 
import pandas as pd
import plotly_express as px
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as num

df = pd.read_csv("cardiacdata.csv")
age  = df["age"]
cardiac = df["target"]

age_train , age_test , cardiac_train , cardiac_test = train_test_split(age, cardiac , test_size = 0.25 , random_state = 0)

x = num.reshape(age_train.ravel(), (len (age_train), 1))
y = num.reshape(cardiac_train.ravel(), (len (cardiac_train), 1))

classifier = LogisticRegression(random_state = 0)
classifier.fit(x , y)

x_test = num.reshape(age_test.ravel(), (len (age_test), 1))
y_test = num.reshape(cardiac_test.ravel(), (len (cardiac_test), 1))

prediction = classifier.predict(x_test)

PredictionValue = []

for i in prediction:
    if i == 0:
        PredictionValue.append("false")
    else:
        PredictionValue.append("true")

ActualValue = []

for i in y_test.ravel():
    if i ==0:
        ActualValue.append("false")
    else: 
        ActualValue.append('true')

labels = ["true" , "false"]
cm = confusion_matrix(ActualValue , PredictionValue)
ax = plt.subplot()
sns.heatmap(cm,annot = True, ax = ax)
ax.set_xlabel('predicted')
ax.set_ylabel('actual')
ax.set_title('CONFUSION MATRIX')
ax.xaxis.set_ticklabels(labels);ax.yaxis.set_ticklabels(labels)