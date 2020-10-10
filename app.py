import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
#TODO: Use model from predict.py
#model = pickle.load(
 #   open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    Main Input For Receiving Query to our ML for the fields,  
    ID	join_date	sex	marital_status	birth_year	branch_code	occupation_code	occupation_category_code	
    + INSURANCE PRODUCT (P5DA	RIBP	8NN1	7POT	66FJ	GYSR	SOP4	RVSZ	PYUQ	LJR9	N2MW	AHXO	BSTQ	FM3X	K6QO	
    QBOL	JWFN	JZ9D	J9JW	GHYX	ECY3) & other feature engineered columns
    '''
    join_date = request.form.get('join_date')
    sex = request.form.get('sex')
    marital_status = request.form.get('marital_status')
    birth_year = int(request.form.get('birth_year'))
    branch_code = request.form.get('branch_code')
    occupation_code = request.form.get('occupation_code')
    occupation_category_code = request.form.get('occupation_category_code')
    insurance_product = request.form.get('insurance_product')

    #TODO: Refactor below

    #prediction = model.predict(
     #   [[join_date, sex, marital_status, birth_year, branch_code, occupation_code, occupation_category_code, INSURANCE PRODUCT]])
    #output = prediction[0]
    output = "prediction[0]"

    return render_template('index.html', prediction_text='The best suited insurance product is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
