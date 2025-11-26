import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data_path = "Lab2\Task1\income_data.txt"
features, labels = [], []
limit_per_class = 25000
count_low_income = count_high_income = 0

with open(data_path, "r") as file:
    for line in file:
        if count_low_income >= limit_per_class and count_high_income >= limit_per_class:
            break
        if "?" in line:
            continue
        row = line.strip().split(", ")
        income_label = row[-1]
        if income_label == "<=50K" and count_low_income < limit_per_class:
            features.append(row[:-1])
            labels.append(income_label)
            count_low_income += 1
        elif income_label == ">50K" and count_high_income < limit_per_class:
            features.append(row[:-1])
            labels.append(income_label)
            count_high_income += 1

features = np.array(features)
labels = np.array(labels)

encoded_features = np.empty(features.shape)
encoders = []

for col_index in range(features.shape[1]):
    column = features[:, col_index]
    try:
        encoded_features[:, col_index] = column.astype(float)
        encoders.append(None)
    except ValueError:
        encoder = preprocessing.LabelEncoder()
        encoded_features[:, col_index] = encoder.fit_transform(column)
        encoders.append(encoder)

encoded_features = encoded_features.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    encoded_features, labels, test_size=0.2, random_state=42
)

model = OneVsOneClassifier(LinearSVC(random_state=0))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred, pos_label=">50K") * 100, 2), "%")
print("Recall:", round(recall_score(y_test, y_pred, pos_label=">50K") * 100, 2), "%")
print("F1-score:", round(f1_score(y_test, y_pred, pos_label=">50K") * 100, 2), "%")

new_sample = [
    '37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
    'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
    '0', '0', '40', 'United-States'
]

encoded_sample = []

for i, value in enumerate(new_sample):
    encoder = encoders[i]
    if encoder is None:
        encoded_sample.append(int(value))
    else:
        if value in encoder.classes_:
            encoded_sample.append(encoder.transform([value])[0])
        else:
            print(f"Попередження: '{value}' не було в навчальних даних для колонки {i}.")
            encoded_sample.append(-1)

encoded_sample = np.array(encoded_sample).reshape(1, -1)
sample_prediction = model.predict(encoded_sample)

print("Prediction for new sample:", sample_prediction[0])