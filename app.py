from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.cluster import KMeans

from sklearn.preprocessing import scale

app = Flask(__name__)
model = pickle.load(open("models/kmeans_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/")
def home():
  return render_template("main.html")

@app.route("/predict", methods=["POST"])
def predict():

  balance = float(request.form["balance"])
  balance_frq = float(request.form["balance_frq"])
  purchases = float(request.form["purchases"])
  oneoff_purchases = float(request.form["oneoff_purchases"])
  installment_purchases = float(request.form["installment_purchases"])
  cash_advance = float(request.form["cash_advance"])
  purchases_frq = float(request.form["purchases_frq"])
  oneoff_purchases_frq = float(request.form["oneoff_purchases_frq"])
  installment_purchases_frq = float(request.form["installment_purchases_frq"])
  cash_advance_frq = float(request.form["cash_advance_frq"])
  cash_advance_trx = float(request.form["cash_advance_trx"])
  purchases_trx = float(request.form["purchases_trx"])
  credit_limit = float(request.form["credit_limit"])
  payments = float(request.form["cash_advance"])
  minimum_payments = float(request.form["minimum_payments"])
  prc_full_payments = float(request.form["prc_full_payments"])
  tenure = float(request.form["tenure"])

  val = [balance, balance_frq, purchases, oneoff_purchases, installment_purchases, cash_advance, purchases_frq, oneoff_purchases_frq, installment_purchases_frq, cash_advance_frq, cash_advance_trx, purchases_trx, credit_limit, payments, minimum_payments, prc_full_payments, tenure]
  val = scaler.transform([val])
  val_predict = model.predict(val)
  return render_template("predict.html", data=val_predict)

  

if __name__ == "__main__":
  app.run(debug=True)
