from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv("data/creditcard.csv")
X = df.drop(columns="Class")
y = df["Class"]

X_train, X_temp, y_train,y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp

)

# print("x train",X_train.shape)
# print("y train",y_train.shape)
# print("x test", X_test.shape)
# print("y test", y_test.shape)
model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
scaler = StandardScaler()
model_pipeline =  Pipeline([
    ("scaler", scaler),
    ("model", model)

])

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_val)

report = classification_report(y_val, y_pred)
# print(report)
cm = confusion_matrix(y_val, y_pred)
# print(cm)
y_probab = model_pipeline.predict_proba(X_val)[:,1]
thresholds = [0.5, 0.7, 0.9]
results = []

for threshold in thresholds:
    custom_pred = (y_probab >= threshold).astype(int)
    new_report = classification_report(y_val, custom_pred)
    new_cm = confusion_matrix(y_val, custom_pred)

    result = f"Threshold: {threshold} Classification Report:{new_report}Confusion Matrix:{new_cm}"

    results.append(result)

# for r in results:
    # print(r)
final_threshold = 0.7
final_probab = model_pipeline.predict_proba(X_test)[:,1]

final_custom_pred = (final_probab >= final_threshold).astype(int)
final_report = classification_report(y_test, final_custom_pred)
final_cm = confusion_matrix(y_test, final_custom_pred)
print(final_report)
print(final_cm)
joblib.dump({"pipeline":model_pipeline, "threshold": final_threshold}, "model/fraud_pipline.pkl")