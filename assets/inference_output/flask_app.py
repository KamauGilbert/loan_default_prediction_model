from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'best_xgb_model_with_class_weights.pkl')
best_xgb_model = joblib.load(model_path)

# Load the preprocessor
preprocessor_path = os.path.join(os.path.dirname(__file__), 'preprocessor.pkl')
preprocessor = joblib.load(preprocessor_path)

@app.route('/')
def welcome():
    return 'Welcome to the Lenders Application Verification System—where your Loan application journey begins!'

@app.route('/predict', methods=['GET'])
def predict_default_status():
    try:
        # Retrieve query parameters
        age = int(request.args.get('age'))
        income = float(request.args.get('income'))
        loanamount = float(request.args.get('loanamount'))
        creditscore = int(request.args.get('creditscore'))
        monthsemployed = int(request.args.get('monthsemployed'))
        numcreditlines = int(request.args.get('numcreditlines'))
        interestrate = float(request.args.get('interestrate'))
        loanterm = int(request.args.get('loanterm'))
        dtiratio = float(request.args.get('dtiratio'))
        education = request.args.get('education')
        employmenttype = request.args.get('employmenttype')
        maritalstatus = request.args.get('maritalstatus')
        hasmortgage = request.args.get('hasmortgage').lower() == 'yes'
        hasdependents = request.args.get('hasdependents').lower() == 'yes'
        loanpurpose = request.args.get('loanpurpose')
        hascosigner = request.args.get('hascosigner').lower() == 'yes'

        # Calculate total payment
        totalpayment = loanamount * (1 + interestrate / 100 * loanterm / 12)

        # Prepare input data as DataFrame
        data = {
            'age': [age], 'income': [income], 'loanamount': [loanamount], 'creditscore': [creditscore],
            'monthsemployed': [monthsemployed], 'numcreditlines': [numcreditlines], 'interestrate': [interestrate],
            'loanterm': [loanterm], 'dtiratio': [dtiratio], 'education': [education],
            'employmenttype': [employmenttype], 'maritalstatus': [maritalstatus],
            'hasmortgage': ['Yes' if hasmortgage else 'No'], 'hasdependents': ['Yes' if hasdependents else 'No'],
            'loanpurpose': [loanpurpose], 'hascosigner': ['Yes' if hascosigner else 'No'], 'totalpayment': [totalpayment]
        }
        df = pd.DataFrame(data)

        # Apply preprocessor transformations
        df_prepared = preprocessor.transform(df)

        # Predict using the model
        probabilities = best_xgb_model.predict_proba(df_prepared)[:, 1]
        prediction = best_xgb_model.predict(df_prepared)[0]

        result = {
            'prediction': 'Defaulter' if prediction == 1 else 'Non-defaulter',
            'probability of defaulting': f'{probabilities[0] * 100:.2f}%'
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run()