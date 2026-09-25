# Machine Learning Final Project: Classical Model vs. Neural Network

## 1. Project Description
This repository contains my final project portfolio for the Machine Learning and applications (Option 2). The objective of this project is to build, evaluate, and fairly compare a traditional machine learning model (Logistic Regression) against a Deep Learning model (Dense Neural Network) on the same real-world tabular dataset, judging whether the added complexity of a neural network is worthwhile.

## 2. Problem Statement
A public agency wants to flag likely higher-income households from survey data to target an outreach programme. The task is a binary classification problem: predicting whether an individual earns more than $50,000 a year (`>50K` = 1) or $50,000 and less (`<=50K` = 0). Because the income classes are imbalanced, the F1 score is used as the primary evaluation metric alongside Accuracy.

## 3. Data Set
The dataset used is the Adult (Census Income) dataset, fetched via OpenML:

Size: 48,842 rows and 14 feature columns (split into 39,073 training rows and 9,769 test rows using a stratified 80/20 split).

Features: A mix of 6 numeric features (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) and 8 categorical features (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`).

Class Balance: Imbalanced target distribution, with 37,155 individuals earning `<=50K` (76.1%) and 11,687 earning `>50K` (23.9% positive rate).

## 4. Method & Workflow
To ensure a fair comparison without data leakage, both models shared the exact same preprocessing pipeline fitted strictly on the training data:
Preprocessing (`ColumnTransformer`):

Numeric features: Missing values imputed using the `median` strategy, followed by standardization (`StandardScaler`).
Categorical features: Missing values imputed using the `most_frequent` strategy, followed by One-Hot Encoding (`OneHotEncoder`), resulting in 105 input features.

Classical Model: A `LogisticRegression` classifier (`max_iter=1000`) wrapped in a scikit-learn `Pipeline`, validated using 5-fold Stratified Cross-Validation on the training set.

Neural Network (Keras): A Sequential Dense network consisting of an Input layer (105 features), a first hidden `Dense` layer (64 units, ReLU) followed by `Dropout(0.3)`, a second hidden `Dense` layer (32 units, ReLU), and a single-unit `sigmoid` output layer. Compiled with the `adam` optimizer and `binary_crossentropy` loss, and trained using `EarlyStopping` (`patience=3`, monitoring `val_loss`).

## 5. Key Results
During 5-fold cross-validation on the training set, Logistic Regression achieved a steady mean F1 score of **0.6576 ± 0.0098**.

On the held-out test set (9,769 samples), the two models performed as follows:

| Model | Test Accuracy | Test F1 Score | True Positives (>50K) | False Negatives | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8524 | 0.6562 | 1,376 | 962 | 480 |
| **Neural Network (Keras)** | **0.8590** | **0.6747** | **1,428** | **910** | **467** |

Logistic Regression confusion matrix:



<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/f6b8454f-70a8-4513-b58a-4d39d594062e" />



Neural Network confusion matrix:



<img width="640" height="480" alt="Figure_3" src="https://github.com/user-attachments/assets/72915c85-3bfc-435c-9ad0-7c4e636f92af" />


Additional graphs:



<img width="800" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/2b9ec20d-f36b-4873-ac32-c8f52ba28ef0" />



<img width="800" height="500" alt="Figure_4" src="https://github.com/user-attachments/assets/ca2156a0-95ce-4c91-8585-ceb43c232d46" />




## 6. Interpretation & Ethical Considerations
Performance Analysis: The Neural Network outperformed Logistic Regression, achieving a higher F1 score (+0.0185) and higher Accuracy (+0.0066). Looking at the confusion matrices, the neural network identified the higher-income minority class more successfully (1,428 vs. 1,376 True Positives) while making fewer False Negative errors (910 vs. 962). Its hidden layers and non-linear activations allowed it to capture complex feature interactions that a strictly linear model cannot map without manual feature engineering.

Why Accuracy is Insufficient: Because 76.1% of individuals earn `<=50K`, a naive model could achieve ~76% accuracy simply by predicting the majority class for everyone. F1 directly measures the balance between precision and recall on the positive minority class (`>50K`).

Ethical Risks & Transparency: Deploying a model trained on demographic attributes (`sex`, `race`, `native-country`) risks reproducing historical biases. In an outreach setting, a False Negative carries a high social cost, as it may exclude a household from receiving needed support. Therefore, despite the Neural Network's slight edge in F1 score, a simpler, transparent model like Logistic Regression is often preferable in public policy so decisions affecting citizens can be audited and explained.

## 7. Reflection
What worked well: Building a unified `ColumnTransformer` preprocessing pipeline ensured zero data leakage and allowed seamless feature transformation for both scikit-learn and Keras. Implementing `Dropout` and `EarlyStopping` effectively prevented the neural network from overfitting, stabilizing validation loss within 6 epochs.

Challenges & Future Improvements: Tuning the neural network architecture on one-hot encoded tabular data required careful configuration. With more time, I would experiment with other classical models to see if i could get a better F1 Score.
