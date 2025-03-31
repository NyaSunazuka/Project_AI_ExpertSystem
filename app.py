from flask import, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# โหลดโมเดลที่เทรนไว้ (ต้องบันทึกโมเดลเป็นไฟล์ .pkl ก่อน)
with open("model.pkl", "rb") as file:
    model, label_encoder = pickle.load(file)

# หน้าหลัก แสดงฟอร์มรับข้อมูลผู้ใช้
@app.route("/")
def index():
    return render_template("index.html")

# รับข้อมูลจากฟอร์ม และทำนายโรค
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # ดึงข้อมูลจากฟอร์ม
        data = [
            int(request.form["Fever"]),
            int(request.form["Cough"]),
            int(request.form["Fatigue"]),
            int(request.form["Difficulty_Breathing"]),
            int(request.form["Age"]),
            int(request.form["Gender"]),
            int(request.form["Blood_Pressure"]),
            int(request.form["Cholesterol_Level"])
        ]
        
        # แปลงเป็น DataFrame และทำนาย
        new_data = pd.DataFrame([data], columns=[
            "Fever", "Cough", "Fatigue", "Difficulty Breathing", "Age",
            "Gender", "Blood Pressure", "Cholesterol Level"
        ])
        prediction = model.predict(new_data)
        predicted_disease = label_encoder.inverse_transform(prediction)[0]
        
        return render_template("result.html", disease=predicted_disease)

if __name__ == "__main__":
    app.run(debug=True)
