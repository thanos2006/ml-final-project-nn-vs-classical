import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)

import keras
from keras import layers

# Logistic Regression

X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)

print("Rows and columns:", X.shape)
print("Target values:", y.value_counts().to_dict())
X.head()

y = LabelEncoder().fit_transform(y)
print("Positive rate (share earning >50K):", round(y.mean(), 3))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Training rows:", X_train.shape[0])
print("Test rows:    ", X_test.shape[0])

numeric_features = X_train.select_dtypes(include="number").columns.tolist()
categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

print("Numeric features:    ", numeric_features)
print("Categorical features:", categorical_features)

numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())])



categorical_pipe = Pipeline ([("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])



preprocess = ColumnTransformer(transformers=[("num", numeric_pipe, numeric_features), ("cat", categorical_pipe, categorical_features)], remainder = "drop" )

clf = Pipeline(steps=[("preprocess", preprocess),("model", LogisticRegression(max_iter=1000))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")
print("cv f1 score for each fold", np.round(cv_scores, 4))
print("mean f1 ± Std: ", np.round(cv_scores.mean(), 4), "±", np.round(cv_scores.std(), 4))

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
clf_acc = accuracy_score(y_test, y_pred)
clf_f1= f1_score(y_test, y_pred)
print("Classical Model (logistic regression) Accuracy:", round(clf_acc, 4))
print("Classical Model (logistic regression) F1 Score:", round(clf_f1, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

X_train_prep = preprocess.fit_transform(X_train)
X_test_prep = preprocess.transform(X_test)

print("Training shape:", X_train_prep.shape)

# Neural network

net = keras.Sequential([
    layers.Input(shape=(X_train_prep.shape[1],)),
    layers.Dense(64, activation= "relu"),
layers.Dropout(0.3),
layers.Dense(32, activation="relu"),
layers.Dense(1, activation="sigmoid")
])
net.summary()

net.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
#fit
history = net.fit(
    X_train_prep, y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=128,
    callbacks=[early_stopping],
    verbose=1
)

print("-" * 100)
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Neural Network Learning Curve (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()
plt.show()

nn_prob = net.predict(X_test_prep).ravel()
nn_pred = (nn_prob > 0.5).astype(int)

nn_acc = accuracy_score(y_test, nn_pred)
nn_f1 = f1_score(y_test, nn_pred)

print("Neural Network Accuracy:", round(nn_acc, 4))
print("Neural Network F1 Score:", round(nn_f1, 4))

ConfusionMatrixDisplay.from_predictions(y_test, nn_pred)
plt.show()

# comparison

print("The comparison (Logistic Regression vs Neural Networks)")
print("Logistic Regression")
print("Accuracy:", round(clf_acc, 4), "| F1 Score:", round(clf_f1, 4))
print("-" * 40)
print("Neural Network")
print("Accuracy:", round(nn_acc, 4), "| F1 Score:", round(nn_f1, 4))

print("-" * 40)

labels = ['Accuracy', 'F1 Score']
clf_scores = [clf_acc, clf_f1]
nn_scores = [nn_acc, nn_f1]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, clf_scores, width, label='Logistic Regression', color='#4C72B0')
rects2 = ax.bar(x + width/2, nn_scores, width, label='Neural Network', color='#C44E52')

ax.set_ylabel('Scores')
ax.set_title('Model Comparison: Logistic Regression vs Neural Network')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([0, 1.0])
ax.legend()

plt.show()

