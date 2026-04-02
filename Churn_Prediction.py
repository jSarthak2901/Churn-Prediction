from google.colab import files
uploaded = files.upload()


import pandas as pd

df=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()

df.shape


df.columns


df['Churn'].value_counts()


df.dtypes


df['TotalCharges'].isnull().sum()


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].isnull().sum()


df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)


df.drop('customerID', axis=1, inplace=True)


X = df.drop('Churn', axis=1)
y = df['Churn']


y = y.map({'Yes': 1, 'No': 0})


X_encoded = pd.get_dummies(X, drop_first=True)


X.shape
X_encoded.shape


X_encoded.isnull().sum().sum()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)


y_pred_dt = dt.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))


import pandas as pd

feature_importance = pd.Series(
    model.coef_[0],
    index=X_encoded.columns
).sort_values(key=abs, ascending=False)

feature_importance.head(10)


tree_importance = pd.Series(
    dt.feature_importances_,
    index=X_encoded.columns
).sort_values(ascending=False)

tree_importance.head(10)


from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
auc


fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.show()


import numpy as np

y_pred_custom = (y_prob >= 0.4).astype(int)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred_custom)


import seaborn as sns
import matplotlib.pyplot as plt


sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()


sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges vs Churn')
plt.show()


sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure vs Churn')
plt.show()


sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Contract Type vs Churn')
plt.show()


sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title('Internet Service vs Churn')
plt.show()


from sklearn.linear_model import LogisticRegression

model_balanced = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model_balanced.fit(X_train, y_train)


y_pred_bal = model_balanced.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred_bal))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bal))
print("\nClassification Report:\n", classification_report(y_test, y_pred_bal))
