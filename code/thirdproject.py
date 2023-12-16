from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv("datasets/train.csv")

# Remove leading and trailing whitespaces from column names
data.columns = data.columns.str.strip()

# Drop unnecessary columns
data.drop(['ID'], axis=1, inplace=True)

# Encode categorical variables
label_encoder = preprocessing.LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = label_encoder.fit_transform(data[col].astype(str))

# Fill missing values with mean
data.fillna(data.mean(), inplace=True)

# Split data into features (X) and target variable (Y)
X = data.drop(['Loan Status'], axis=1)
Y = data['Loan Status']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.4, random_state=1)

# Initialize classifiers
knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(
    n_estimators=7, criterion='entropy', random_state=7)
lc = LogisticRegression(solver='lbfgs', max_iter=70000)
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
# Create lists to store the results
algorithms = []
accuracy_values = []
precision_values = []
recall_values = []
f1_values = []
print('making predictions on the testing set')
for clf in (rfc, knn, nb, lc, dtc, gbc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred,zero_division=1)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred,average='weighted')

    # Store values in lists
    algorithms.append(algoName[clf])
    accuracy_values.append(accuracy)
    precision_values.append(precision)
    recall_values.append(recall)
    f1_values.append(f1)

    print(f"Metrics for {clf.__class__.__name__} on testing set:")
    print("Accuracy =", accuracy)
    print("Precision =", precision)
    print("Recall =", recall)
    print("F1-Score =", f1)
    print("-------------------------")


# Loop through the algorithms and store the testing set metric values

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