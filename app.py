import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

# Load the data
genes = pd.read_csv("sfari_genes.csv")

# Drop unnecessary columns
columns_to_drop = ['status', 'chromosome', 'number-of-reports', 'gene-name', 'ensembl-id', 'gene-score', 'genetic-category']
genes = genes.drop(columns=columns_to_drop)

# Encode gene symbols as dummy variables
genes_encoded = pd.get_dummies(genes, columns=['gene-symbol'])

# Features (X) excluding the 'syndromic' column
X = genes_encoded.drop(columns='syndromic')

# Labels (y)
y = genes_encoded['syndromic']

# Convert to binary classification (1 for syndromic, 0 for non-syndromic)
y_binary = (y == 1).astype(int)

# Resample the dataset
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y_binary)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize the classifiers
classifiers = {
    'XGBoost': XGBClassifier(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# Create a Streamlit app
st.title("Autism gene Prediction App")

# User input for gene symbol
user_gene_symbol = st.text_input("Enter a gene symbol:")

# Display user input
st.write("You entered:", user_gene_symbol)

# Train and evaluate each classifier on the resampled data
for clf_name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train, y_train)

    # Prepare the input data for prediction
    user_input = pd.get_dummies(pd.Series([user_gene_symbol]), prefix='gene-symbol').reindex(columns=X.columns, fill_value=0)

    # Make prediction
    prediction = clf.predict(user_input)

    # Display prediction
    st.write(f"{clf_name} predicts: {'it is autism gene ' if prediction == 1 else 'it is non-autism gene'}")

    # If you want to display detailed classification report
    if st.checkbox(f"Show Classification Report for {clf_name}"):
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred)
        st.text(f"Classification Report for {clf_name}:\n{report}")
