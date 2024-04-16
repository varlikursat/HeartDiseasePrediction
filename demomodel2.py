import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score


data = pd.read_csv('HeartDiseaseDataset.csv')

# Kategorik olan dataları convert ettiğim kısım
data = pd.get_dummies(data, columns=['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Datasetimizin %60 ını training datası, %20lik kısmını test datası kalan %20lik kısmınsa validation datası olarak ayarladım
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Hyperparametreleri tanımlayıp ayarladığım kısım
param_grid = {
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Decision Tree algoritmamızı dahil ettiğim kısım
dt_classifier = DecisionTreeClassifier(random_state=42)


grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)



y_val_pred = grid_search.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)


y_test_pred = grid_search.predict(X_test)


test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)


print("\nClassification Report For Test Set:")
print(classification_report(y_test, y_test_pred))
