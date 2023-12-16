from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np

# Load data
# Update with the correct file name
data = pd.read_csv("datasets/loan_approval_dataset.csv")

# Print column names
# print("Original Column Names:", data.columns)

# Remove leading and trailing whitespaces from column names
data.columns = data.columns.str.strip()

# Drop unnecessary columns
data.drop(['loan_id'], axis=1, inplace=True)

# For categorical columns, fill missing values with the most frequent value
for col in data.select_dtypes(include='object').columns:
    data[col].fillna(data[col].mode()[0], inplace=True)


# Encode categorical variables
label_encoder = preprocessing.LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = label_encoder.fit_transform(
        data[col].astype(str))  # Ensure data is treated as string

# Fill missing values with mean
data.fillna(data.mean(), inplace=True)

# Split data into features (X) and target variable (Y)
X = data.drop(['loan_status'], axis=1)
Y = data['loan_status']
X.shape, Y.shape

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.4, random_state=1)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

# Initialize classifiers
knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)
lc = LogisticRegression(solver='lbfgs', max_iter=1000)
dtc = DecisionTreeClassifier(random_state=7)
gbc = GradientBoostingClassifier(random_state=7)
nb = GaussianNB()  



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
    algorithms.append((clf.__class__.__name__).replace("Classifier", ""))
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