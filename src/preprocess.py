import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath)
    
    # Drop unnecessary columns
    data = data.drop(columns=['customerID'])
    
    # Convert categorical variables to dummy variables
    data = pd.get_dummies(data, drop_first=True)
    
    # Split features and target
    X = data.drop('Churn_Yes', axis=1)
    y = data['Churn_Yes']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

