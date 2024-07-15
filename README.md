## 📊 **Data Preprocessing**

🔄 **Modeling**

### Data Preprocessing Steps:
1. **Splitting Data**:
   - Divided the dataset into independent columns `X = df.drop('default', axis=1)` and the dependent column `y = df['default']`.
   - Used scikit-learn to split the dataset into 70% train and 30% test:
     ```python
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
     ```

2. **Encoding and Scaling**:
   - Utilized `TargetEncoder` to encode categorical variables and `MinMaxScaler` to scale numerical variables:
     ```python
     from sklearn.compose import ColumnTransformer
     from category_encoders import TargetEncoder
     from sklearn.preprocessing import MinMaxScaler

     categorical_variables = ['education', 'employmenttype', 'maritalstatus', 'hasmortgage', 'hasdependents', 'loanpurpose', 'hascosigner']
     numerical_variables = ['age', 'income', 'loanamount', 'creditscore', 'monthsemployed', 'numcreditlines', 'interestrate', 'loanterm', 'dtiratio']

     preprocessor = ColumnTransformer(transformers=[
         ('te', TargetEncoder(min_samples_leaf=1, smoothing=10), categorical_variables),
         ('scaler', MinMaxScaler(), numerical_variables)
     ], remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")
     ```

3. **Saving the Preprocessor**:
   - Saved the preprocessor to a pickle file for use during inference:
     ```python
     import pickle

     with open('preprocessor.pkl', 'wb') as file:
         pickle.dump(preprocessor, file)
     ```

By following these preprocessing steps, the dataset is ready for building robust and scalable machine learning models. 🎯
