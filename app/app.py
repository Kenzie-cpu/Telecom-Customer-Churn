import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

model = joblib.load("./app/KNN.pkl")


@app.route("/")
def home():
    return render_template("index.html")


# qn_order is a list of question category in the order of form values returned
qn_order = ["SeniorCitizen", "Partner", "Dependents", "tenure", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "MonthlyCharges", "TotalCharges", "InternetService", "Contract", "PaymentMethod"]

# order of columns that needs to be input into predict function
col_pred = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges',
            'TotalCharges', 'InternetService_DSL',
            'InternetService_Fiber optic', 'InternetService_No',
            'Contract_Month-to-month', 'Contract_One year',
            'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
            'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

ohe_dict = {"InternetService": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "Contract": [[1, 0, 0], [
    0, 1, 0], [0, 0, 1]], "PaymentMethod": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}


@app.route("/predict", methods=["POST"])
def predict():
    print("predict route invoked")
    form_data = [x for x in request.form.values()]

    print(form_data, len(form_data), len(qn_order))
    # handle label-encoding, load from npy files for persistent label encoding with training/testing data
    encoder = LabelEncoder()
    for i in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'MultipleLines', 'StreamingMovies', 'Partner', 'Dependents', 'PaperlessBilling', "SeniorCitizen"]:
        encoder.classes_ = np.load(
            f'./app/encoder_exports/{i}.npy', allow_pickle=True)
        index = qn_order.index(i)
        arr = encoder.transform(
            np.array(form_data[index]).reshape(-1, 1))
        form_data[index] = arr[0]

    form_data = [int(x) for x in form_data]

    # handling one-hot encoded values
    ohe_col = ["InternetService", "Contract", "PaymentMethod"]
    counter = 0
    for i in range(-3, 0):
        form_data.extend(ohe_dict[ohe_col[counter]]
                         [form_data.pop(i)])  # append to list
        print(form_data)
        counter += 1

    features = [np.array(form_data)]
    prediction = model.predict(features)
    if prediction == 1:
        prediction = "This customer is more likely to churn"
    else:
        prediction = "This customer is less likely to churn"

    print(form_data)
    return render_template("index.html", churn_prediction=prediction)


if __name__ == "__main__":
    app.run()
