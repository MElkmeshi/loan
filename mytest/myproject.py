from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
data = pd.read_csv("loan/mytest/train.csv")

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
svc = SVC()
lc = LogisticRegression(solver='lbfgs', max_iter=70000)
dtc = DecisionTreeClassifier(random_state=7)
gbc = GradientBoostingClassifier(random_state=7)

print('making predictions on the training set')
for clf in (rfc, knn, svc, lc, dtc, gbc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_train)

    accuracy = accuracy_score(Y_train, Y_pred)
    precision = precision_score(Y_train, Y_pred, zero_division=1)
    recall = recall_score(Y_train, Y_pred, zero_division=1)
    f1 = f1_score(Y_train, Y_pred, zero_division=1)

    print(f"Metrics for {clf.__class__.__name__} on training set:")
    print("Accuracy =", accuracy)
    print("Precision =", precision)
    print("Recall =", recall)
    print("F1-Score =", f1)
    print("-------------------------")

print('making predictions on the testing set')
for clf in (rfc, knn, svc, lc, dtc, gbc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, zero_division=1)
    recall = recall_score(Y_test, Y_pred, zero_division=1)
    f1 = f1_score(Y_test, Y_pred, zero_division=1)

    print(f"Metrics for {clf.__class__.__name__} on testing set:")
    print("Accuracy =", accuracy)
    print("Precision =", precision)
    print("Recall =", recall)
    print("F1-Score =", f1)
    print("-------------------------")
