from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

data = pd.read_csv("LoanApprovalPrediction.csv")

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
                                                    test_size=0.4,
                                                    random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(n_estimators=7,
                             criterion='entropy',
                             random_state=7)
svc = SVC()
lc = LogisticRegression(solver='lbfgs', max_iter=1000)
dtc = DecisionTreeClassifier(random_state=7)
gbc = GradientBoostingClassifier(random_state=7)

print('making predictions on the training set')
for clf in (rfc, knn, svc, lc, dtc, gbc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_train)

    accuracy = accuracy_score(Y_train, Y_pred)
    precision = precision_score(Y_train, Y_pred)
    recall = recall_score(Y_train, Y_pred)
    f1 = f1_score(Y_train, Y_pred)

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
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)

    print(f"Metrics for {clf.__class__.__name__} on testing set:")
    print("Accuracy =", accuracy)
    print("Precision =", precision)
    print("Recall =", recall)
    print("F1-Score =", f1)
    print("-------------------------")
