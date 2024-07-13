### 🚀 Models & Pre-Processor Subfolder Overview 📊

Welcome to the Models & Pre-Processor subfolder! This section houses the top-performing XGBoost models 🌟: one trained with class weights applied and another without. Additionally, you'll find a pickle file 📦 containing the preprocessor for both numerical and categorical variables, ensuring seamless data preprocessing. A notebook 📓 is also included, providing a guide on how to use the models.

#### Inferencing Guidelines 📏

When making predictions, please follow these rules:

- **Age**: Use an integer, not a float
- **Income**: Use either an integer or a float
- **Loan Amount**: Use either an integer or a float
- **Credit Score**: Use integers between 300 and 850
- **Months Employed**: Use integers
- **Number of Credit Lines**: Use integers
- **Interest Rate**: Use integers or floats (will be converted to a percentage)
- **Loan Term**: Loan period in months, use integers
- **DTI Ratio**: Use floats between 0 to 1 (ratio)
- **Education**: Choose between High School, Bachelor's, Master's, or PhD (exact match)
- **Employment Type**: Choose between Unemployed, Self-employed, Part-time, or Full-time (exact match)
- **Marital Status**: Choose between Single, Divorced, or Married
- **Has Mortgage**: Yes/No
- **Has Dependents**: Yes/No
- **Loan Purpose**: Choose between Auto, Business, Education, Home, or Other (exact match)
- **Has Cosigner**: Yes/No

#### Package Requirements 📦

Before inferencing, ensure you have the necessary packages, such as `category_encoders`. If not, install it using:

- **Google Colab**: `!pip install category_encoders`
- **VSCode Terminal or other IDEs**: `pip install category_encoders`

Happy predicting! 🎉
