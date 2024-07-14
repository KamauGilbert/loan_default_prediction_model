import os
import pandas as pd
import joblib
import streamlit as st

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'best_xgb_model_with_class_weights.pkl')
best_xgb_model = joblib.load(model_path)

# Load the preprocessor
preprocessor_path = os.path.join(os.path.dirname(__file__), 'preprocessor.pkl')
preprocessor = joblib.load(preprocessor_path)

# Streamlit UI
def main():
    st.image('K.GI Lenders Logo.jpg', use_column_width=True)
    html_temp = """
    <div style="background-color: green; padding: 10px">
    <h2 style="color: white; text-align: center;">Loan Default Prediction App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.write(welcome())

    # User inputs
    age = st.text_input('Age (integer)')
    income = st.text_input('Income (integer/float)')
    loanamount = st.text_input('Loan Amount (integer/float)')
    creditscore = st.text_input('Credit Score (between 300-850)')
    monthsemployed = st.text_input('Months Employed (integer)')
    numcreditlines = st.text_input('Number of Credit Lines (integer)')
    interestrate = st.text_input('Interest Rate (integer/float between 0-100)')
    loanterm = st.text_input('Loan Term in months (integer)')
    dtiratio = st.text_input('DTI Ratio (between 0-1)')
    education = st.selectbox('Education Level', ["High School", "Bachelor's", "Master's", "PhD"])
    employmenttype = st.selectbox('Employment Type', ['Full-Time', 'Part-Time', 'Self-Employed', 'Unemployed'])
    maritalstatus = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    hasmortgage = st.selectbox('Has Mortgage', ['Yes', 'No'])
    hasdependents = st.selectbox('Has Dependents', ['Yes', 'No'])
    loanpurpose = st.selectbox('Loan Purpose', ['Auto', 'Business', 'Education', 'Home', 'Other'])
    hascosigner = st.selectbox('Has Co-signer', ['Yes', 'No'])

    # Prediction button
    if st.button('Predict'):
        try:
            # Convert inputs to appropriate types
            age = int(age)
            income = float(income)
            loanamount = float(loanamount)
            creditscore = int(creditscore)
            monthsemployed = int(monthsemployed)
            numcreditlines = int(numcreditlines)
            interestrate = float(interestrate)
            loanterm = int(loanterm)
            dtiratio = float(dtiratio)

            # Calculate total payment
            totalpayment = loanamount * (1 + interestrate / 100 * loanterm / 12)

            # Prepare input data as DataFrame
            data = {
                'age': [age], 'income': [income], 'loanamount': [loanamount], 'creditscore': [creditscore],
                'monthsemployed': [monthsemployed], 'numcreditlines': [numcreditlines], 'interestrate': [interestrate],
                'loanterm': [loanterm], 'dtiratio': [dtiratio], 'education': [education],
                'employmenttype': [employmenttype], 'maritalstatus': [maritalstatus],
                'hasmortgage': ['Yes' if hasmortgage == 'Yes' else 'No'], 
                'hasdependents': ['Yes' if hasdependents == 'Yes' else 'No'],
                'loanpurpose': [loanpurpose], 
                'hascosigner': ['Yes' if hascosigner == 'Yes' else 'No'], 
                'totalpayment': [totalpayment]
            }
            df = pd.DataFrame(data)

            # Apply preprocessor transformations
            df_prepared = preprocessor.transform(df)

            # Predict using the model
            probabilities = best_xgb_model.predict_proba(df_prepared)[:, 1]
            prediction = best_xgb_model.predict(df_prepared)[0]

            # Display result
            result = {
                'prediction': 'Defaulter' if prediction == 1 else 'Non-defaulter',
                'probability': f'{probabilities[0] * 100:.2f}%'
            }
            st.subheader('Prediction Result')
            st.write(f"Prediction: {result['prediction']}")
            st.write(f"Probability of Defaulting: {result['probability']}")

        except ValueError as e:
            st.error(f"Invalid input: {e}")

def welcome():
    return 'Welcome to the Lenders Application Verification System—where your Loan application journey begins!'

if __name__ == '__main__':
    main()