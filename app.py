from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)


mul = open('gbc_scale_flask.pkl', 'rb')
ml_model = pickle.load(mul)

scale_mul = open('scale_flask.pkl', 'rb')
scaler = pickle.load(scale_mul)


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("I was here 1")
    if request.method == 'POST':
        print(request.form.get('TargetAge'))
        try:
            TargetAge = float(request.form['TargetAge'])
            HousingIndex = float(request.form['HousingIndex'])
            ppiscore = float(request.form['ppiscore'])
            peopleAgesBelow17 = float(request.form['peopleAgesBelow17'])
            LoanCycle = float(request.form['LoanCycle'])
            CBR_CUR_BAL_HM = float(request.form['CBR_CUR_BAL_HM'])
            LoanAmountApproved = float(request.form['LoanAmountApproved'])
            TotalMonthlyExp = float(request.form['TotalMonthlyExp'])
            pred_args = [TargetAge,HousingIndex,ppiscore,peopleAgesBelow17,LoanCycle,CBR_CUR_BAL_HM,LoanAmountApproved,TotalMonthlyExp]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)

            np.set_printoptions(formatter={'float_kind':'{:f}'.format})

            pred_args_arr = scaler.transform(pred_args_arr)

            model_prediction = ml_model.predict(pred_args_arr)
            model_probability= ml_model.predict_proba(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction = model_prediction, normal_probability='{:.2%}'.format(model_probability[0,0]), risky_probability='{:.2%}'.format(model_probability[0,1]))


if __name__ == "__main__":
    app.run(host='0.0.0.0')
