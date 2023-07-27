from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

import pandas as pd

df = pd.read_csv("preprocessed_new_dataset.csv")

x = df.drop(['lesion_type', "image_id", "confirmation"], axis = 1)
y = df['lesion_type']

ros = RandomOverSampler(sampling_strategy = 'auto', random_state = 0)

x, y = ros.fit_resample(x,y)

X_train, X_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size = 0.2, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

dtc = DecisionTreeClassifier(random_state=42)
scores = cross_val_score(DecisionTreeClassifier(), x, y, cv=5)
scores 

accuracy_dtc = cross_val_score(dtc, x, y, cv=5, scoring='accuracy')
precision_dtc = cross_val_score(dtc, x, y, cv=5, scoring='precision_weighted')
recall_dtc = cross_val_score(dtc, x, y, cv=5, scoring='recall_weighted')
f1_dtc = cross_val_score(dtc, x, y, cv=5, scoring='f1_weighted')

print('Cross-Validation of Performance Decision Tree')
print('Accuracy', accuracy_dtc)
print('Precision', precision_dtc)
print('Recall', recall_dtc)
print('F1-Score', f1_dtc)