# 📊 **Inferencing Subfolder**

Welcome to the inferencing subfolder! Here, you'll find everything you need to make predictions using our models.

### 📓 **Notebook Guide**
A detailed notebook is included, providing a step-by-step guide on how to use the models. Dive in and explore!

---

## 📏 **Inferencing Guidelines**

When making predictions, please follow these rules:

- **Preprocessor & Models**: Input the path to the preprocessor and either of the XGB models as directed in the notebook.
- **Age**: Use an integer, not a float.
- **Income**: Use either an integer or a float.
- **Loan Amount**: Use either an integer or a float.
- **Credit Score**: Use integers between 300 and 850.
- **Months Employed**: Use integers.
- **Number of Credit Lines**: Use integers.
- **Interest Rate**: Use integers or floats (will be converted to a percentage).
- **Loan Term**: Loan period in months, use integers.
- **DTI Ratio**: Use floats between 0 to 1 (ratio).
- **Education**: Choose between High School, Bachelor's, Master's, or PhD (exact match).
- **Employment Type**: Choose between Unemployed, Self-employed, Part-time, or Full-time (exact match).
- **Marital Status**: Choose between Single, Divorced, or Married.
- **Has Mortgage**: Yes/No.
- **Has Dependents**: Yes/No.
- **Loan Purpose**: Choose between Auto, Business, Education, Home, or Other (exact match).
- **Has Cosigner**: Yes/No.

---

## 📦 **Package Requirements**

Before inferencing, ensure you have the necessary packages, such as `category_encoders`. If not, install it using:

- **Google Colab**: `!pip install category_encoders`
- **VSCode Terminal or other IDEs**: `pip install category_encoders`

### 🛠️ **Setup Instructions**

1. **Virtual Environment**: Open the terminal on your IDE, create a virtual environment, and satisfy the `requirements.txt` file. Ensure you have Python 3.12.2.
2. **Run the Application**:
   - **Streamlit**: Change the directory to the folder where the Streamlit file is located and run:
     ```bash
     streamlit run streamlit.py
     ```
   - **Flask**: To run the Flask application, follow these steps:
     1. Navigate to the directory where the Flask application file is located.
     2. Run the Flask application with:
        ```bash
        python flask_app.py
        ```
     3. The terminal will display the local server address (e.g., `http://127.0.0.1:5000/`). Copy this address.
     4. Open Postman, create a new request, and paste the copied address into the request URL field to test the endpoints.


Let's make some accurate predictions! 🚀
