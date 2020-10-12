import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from predict import predict_product
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    join_date = request.form.get('join_date')
    sex = request.form.get('sex')
    marital_status = request.form.get('marital_status')
    birth_year = int(request.form.get('birth_year'))
    branch_code = request.form.get('branch_code')
    occupation_code = request.form.get('occupation_code')
    occupation_category_code = request.form.get('occupation_category_code')
    insurance_product = request.form.get('insurance_product')

    products_list = ['P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 
      'N2MW', 'AHXO','BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 
      'ECY3']

    data = {
        'ID': 'XY234ZF',
        'join_date': request.form.get('join_date'),
        'sex': request.form.get('sex'),
        'marital_status': request.form.get('marital_status'),
        'birth_year': int(request.form.get('birth_year')),
        'branch_code': request.form.get('branch_code'),
        'occupation_code': request.form.get('occupation_code'),
        'occupation_category_code': request.form.get('occupation_category_code')
    }

    for p in products_list:
        data[p] = 1 if p in insurance_product else 0

    sample = pd.DataFrame([data])
    resp = predict_product(sample)

    output = resp.idxmax(axis=1)[0]
    score = round(resp[output][0], 3)

    return render_template('index.html', prediction_text='The best suited insurance product is {} with confidence {}'.format(output, score))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
