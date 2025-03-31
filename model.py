import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# โหลดข้อมูล
file_path = "Disease_symptom_and_patient_profile_dataset.csv"
df = pd.read_csv(file_path)

# แปลงข้อมูล
yes_no_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
df[yes_no_cols] = df[yes_no_cols].apply(lambda x: x.map({"Yes": 1, "No": 0}))
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

ordinal_cols = ["Blood Pressure", "Cholesterol Level"]
encoder = OrdinalEncoder(categories=[["Low", "Normal", "High"], ["Low", "Normal", "High"]])
df[ordinal_cols] = encoder.fit_transform(df[ordinal_cols])

# แยก Features และ Target
X = df.drop(["Disease", "Outcome Variable"], axis=1)
y = df["Disease"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=1)

# ฝึกโมเดล
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# บันทึกโมเดลและ label encoder
with open("model.pkl", "wb") as file:
    pickle.dump((model, label_encoder), file)

print("Model saved as model.pkl")
