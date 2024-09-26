import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the saved model
filename = 'voting_clf_model.sav'
voting_clf = pickle.load(open(filename, 'rb'))

filename_adaboost = 'adaboost_clf_model.sav'
adaboost_clf = pickle.load(open(filename_adaboost, 'rb'))


st.title("Loan Default Prediction App")

# Create input fields for user to enter data
loan_amount = st.number_input("Loan Amount", min_value=0)
term = st.number_input("Term", min_value=0)
interest_rate = st.number_input("Interest Rate", min_value=0.0)
credit_score = st.number_input("Credit Score", min_value=0, max_value=850)
employment_length = st.number_input("Employment Length (in years)", min_value=0)

loan_purpose_options = ['Debt Consolidation', 'Home Improvement', 'Business', 'Personal', 'Medical', 'Education', 'Auto', 'Other']
loan_purpose = st.selectbox("Loan Purpose", loan_purpose_options)

marital_status_options = ['Married', 'Single', 'Divorced', 'Widowed', 'Separated']
marital_status = st.selectbox("Marital Status", marital_status_options)

# Load the label encoder used during training
le_loan_purpose = LabelEncoder()
le_marital_status = LabelEncoder()

# Fit the label encoder with the training data if you have it saved
# Otherwise, you can use the options above to encode the input

le_loan_purpose.fit(loan_purpose_options)
le_marital_status.fit(marital_status_options)

# Encode the user input using the label encoder
encoded_loan_purpose = le_loan_purpose.transform([loan_purpose])[0]
encoded_marital_status = le_marital_status.transform([marital_status])[0]


# Create a dataframe with user input
new_data = pd.DataFrame({
    'Loan_Amount': [loan_amount],
    'Term': [term],
    'Interest_Rate': [interest_rate],
    'Credit_Score': [credit_score],
    'Employment_Length': [employment_length],
    'Loan_Purpose': [encoded_loan_purpose],
    'Marital_Status': [encoded_marital_status]
})


# Make predictions using the loaded model
if st.button("Predict Loan Default"):
    prediction_voting = voting_clf.predict(new_data)
    prediction_adaboost = adaboost_clf.predict(new_data)

    if prediction_voting[0] == 1:
        st.write("Voting Classifier Prediction: **Loan Default is likely**")
    else:
        st.write("Voting Classifier Prediction: **Loan Default is unlikely**")

    if prediction_adaboost[0] == 1:
        st.write("AdaBoost Classifier Prediction: **Loan Default is likely**")
    else:
        st.write("AdaBoost Classifier Prediction: **Loan Default is unlikely**")
