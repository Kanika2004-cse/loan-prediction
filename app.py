from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    income = float(request.form['income'])
    loan = float(request.form['loan'])
    credit = float(request.form['credit'])
    education = int(request.form['education'])
    married = int(request.form['married'])

    features = np.array([[income, loan, credit, education, married]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "✅ Loan Approved"
    else:
        result = "❌ Loan Rejected"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
