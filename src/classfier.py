import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('stocks_enhanced.csv')

features = df.iloc[:, 1:4].values
target = df.iloc[:, 6].values

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25)

model = RandomForestClassifier(n_estimators = 100, random_state=0)

model.fit(features_train, target_train)

prediction_train = model.predict(features_train)
prediction_test = model.predict(features_test)

accuracy_train = accuracy_score(target_train, prediction_train)
print('Accuracy of the model on training data set', accuracy_train)

accuracy_test = accuracy_score(target_test, prediction_test)
print('Accuracy of the model on test data set', accuracy_test)

print('Actual values:', target_test)
print('Predicted values:', prediction_test)

#test an example
def predict_stock(pe, price, eps):
    example = [[pe, price, eps]]
    return model.predict(example)[0]