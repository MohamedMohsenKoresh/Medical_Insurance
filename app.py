from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
from io import BytesIO

app = Flask(__name__)
model = joblib.load("model.pkl")

memory_file = None  # لتخزين آخر ملف CSV متوقع

@app.route("/", methods=["GET", "POST"])
def index():
    global memory_file
    predictions = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            # قراءة الملف
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            # التنبؤ
            df["Predicted Charges"] = model.predict(df)
            predictions = df

            # حفظ النتائج في الذاكرة للتحميل
            output = BytesIO()
            df.to_csv(output, index=False)
            output.seek(0)
            memory_file = output

    return render_template("index.html", predictions=predictions)

@app.route("/download")
def download():
    global memory_file
    if memory_file:
        memory_file.seek(0)
        return send_file(memory_file, mimetype="text/csv", as_attachment=True, download_name="predictions.csv")
    else:
        return "No file to download", 400

if __name__ == "__main__":
    app.run(debug=True)


