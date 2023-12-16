import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

data = pd.read_csv("datasets/LoanApprovalPrediction.csv")

obj = (data.dtypes == 'object')

data.drop(['Loan_ID'], axis=1, inplace=True)
label_encoder = preprocessing.LabelEncoder()
for col in data.columns:
    if col == "Loan_ID":
        continue
    data[col] = label_encoder.fit_transform(data[col])
    data[col] = data[col].fillna(data[col].mean())
X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.3,
                                                    random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(n_estimators=7,
                             criterion='entropy',
                             random_state=7)
lc = LogisticRegression(solver='lbfgs', max_iter=1000)
dtc = DecisionTreeClassifier(random_state=7)
gbc = GradientBoostingClassifier(random_state=7)
nb = MultinomialNB()

algoName = {
    rfc: "Random Forest",
    knn: "k-NN",
    nb: "Naïve Bayes",
    lc: "Logistic Regression",
    dtc: "Decision Tree",
    gbc: "Gradient Boosting"
}


print('making predictions on the testing set')
for clf in (rfc, knn, nb, lc, dtc, gbc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)

    print(f"Metrics for {clf.__class__.__name__} on testing set:")
    print("Accuracy =", accuracy)
    print("Precision =", precision)
    print("Recall =", recall)
    print("F1-Score =", f1)
    print("-------------------------")

# Create lists to store the results
algorithms = []
accuracy_values = []
precision_values = []
recall_values = []
f1_values = []

# Loop through the algorithms and store the testing set metric values
for clf in (rfc, knn, nb, lc, dtc, gbc):
    clf.fit(X_train, Y_train)
    Y_pred_test = clf.predict(X_test)

    # Calculate metrics for testing set
    accuracy_test = accuracy_score(Y_test, Y_pred_test)
    precision_test = precision_score(Y_test, Y_pred_test)
    recall_test = recall_score(Y_test, Y_pred_test)
    f1_test = f1_score(Y_test, Y_pred_test)

    # Store values in lists
    algorithms.append(algoName[clf])
    accuracy_values.append(accuracy_test)
    precision_values.append(precision_test)
    recall_values.append(recall_test)
    f1_values.append(f1_test)

# Plotting with a horizontal bar chart
width = 0.2
ind = np.arange(len(algorithms))

fig, ax = plt.subplots(figsize=(12, 6))

# Plot Accuracy
rects_acc_test = ax.bar(ind, accuracy_values, width,
                        label='Accuracy', color='skyblue', edgecolor='black', hatch='//')

# Plot Precision
rects_prec_test = ax.bar(ind + width, precision_values,
                         width, label='Precision', color='lightcoral', edgecolor='black', hatch='//')

# Plot Recall
rects_recall_test = ax.bar(
    ind + 2 * width, recall_values, width, label='Recall', color='lightgrey', edgecolor='black', hatch='//')

# Plot F1-Score
rects_f1_test = ax.bar(ind + 3 * width, f1_values, width,
                       label='F1-Score', color='gold', edgecolor='black', hatch='//')

# Add labels, title, and legend
ax.set_xticks(ind + 1.5 * width)
ax.set_xticklabels(algorithms)
ax.set_ylabel('Metrics')
ax.set_title('Testing Set Metrics for Different Algorithms')
ax.legend()

# Display the bar chart
plt.show()
