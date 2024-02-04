import streamlit as st
import pandas as pd
pip install --upgrade pandas
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler
# For Windows
python -m venv venv

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

# Initialize the classifier
classifiers = {
    'XGBoost': XGBClassifier(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier()
}

# Streamlit app
st.title("Autism Gene Predictor")

# Sidebar for user input
gene_symbol = st.sidebar.text_input("Enter a gene symbol:")

# Check if the gene symbol exists in the data
if gene_symbol in genes['gene-symbol'].values:
    # Extract the corresponding row from the dataframe
    gene_info = genes[genes['gene-symbol'] == gene_symbol]

    # Check if the gene is syndromic or not
    if gene_info['syndromic'].values[0] == 1:
        st.sidebar.success(f"The gene {gene_symbol} is associated with autism.")
    else:
        st.sidebar.info(f"The gene {gene_symbol} is not associated with autism.")
else:
    st.sidebar.warning("The gene symbol does not exist in the data.")

# Train and evaluate each classifier on the resampled data
for clf_name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Display results
    st.write(f"\nResults for {clf_name} on resampled data:")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")

    # Classification Report
    report = classification_report(y_test, y_pred)
    st.write(f"Classification Report for {clf_name} on resampled data:\n{report}")
