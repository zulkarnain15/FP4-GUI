from crypt import methods
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('models/kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('models/regression.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("main.html")

@app.route("/predict", methods=['POST'])
def predict():
    balance = float(request.form['balance'])
    balance_frq = float(request.form['balance_frq'])
    purchases = float(request.form['purchases'])
    oneoff_purchases = float(request.form['oneoff_purchases'])
    installment_purchases = float(request.form['installment_purchases'])
    cash_advance = float(request.form['cash_advance'])
    purchases_frq = float(request.form['purchases_frq'])
    oneoff_purchases_frq = float(request.form['oneoff_purchases_frq'])
    installment_purchases_frq = float(request.form['installment_purchases_frq'])
    cash_advance_frq = float(request.form['cash_advance_frq'])
    cash_advance_trx = float(request.form['cash_advance_trx'])
    purchases_trx = float(request.form['purchases_trx'])
    credit_limit = float(request.form['credit_limit'])
    payments = float(request.form['payments'])
    minimum_payments = float(request.form['minimum_payments'])
    prc_full_payments = float(request.form['prc_full_payments'])
    tenure = float(request.form['tenure'])

    val = [balance, balance_frq, purchases, oneoff_purchases, installment_purchases, 
        cash_advance, purchases_frq, oneoff_purchases_frq, installment_purchases_frq,
        cash_advance_frq, cash_advance_trx,
        purchases_trx, credit_limit, payments, minimum_payments, prc_full_payments, tenure]
    val = scaler.transform([val])

    val_predict = model.predict(val)

    if val_predict == 0:
        output = 'Pengguna termasuk ke dalam klaster {}, dengan karakteristik\nPengguna dengan limit kredit kecil dan melakukan transaksi pembayaran dan pembelian cukup sering'.format(val_predict)
    elif val_predict == 1:
        output = 'Pengguna termasuk ke dalam klaster {}, dengan karakteristik\nPengguna dengan limit kredit rata-rata dan tidak banyak melakukan transaksi maupun tarik tunai kartu kredit'.format(val_predict)
    elif val_predict == 2:
        output = 'Pengguna termasuk ke dalam klaster {}, dengan karakteristik\nPengguna dengan limit kredit tinggi dan menggunakannya untuk transaksi tarik tunai'.format(val_predict)
    elif val_predict == 3:
        output = 'Pengguna termasuk ke dalam klaster {}, dengan karakteristik\nPengguna dengan limit kredit paling tinggi dan sering melakukan transaksi baik itu pembelian dan pembayaran ataupun tarik tunai'.format(val_predict)
    elif val_predict == 4:
        output = 'Pengguna termasuk ke dalam klaster {}, dengan karakteristik\nPengguna dengan limit kredit tinggi dengan saldo rata-rata dan menggunakannya untuk transaksi pembelian dan pembayaran'.format(val_predict)
    else:
        output = 'Pengguna tidak termasuk dalam klaster manapun'

    return render_template('main.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)