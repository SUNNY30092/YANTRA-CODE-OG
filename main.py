from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn
from io import StringIO
from fastapi.responses import StreamingResponse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def Fill_missing_values(df):
    print("Original Data with Missing Values:\n", df)

    # df.replace(r'^/s*$', np.nan, regex=True, inplace=True)

    # Function to check if a string column is categorical
    def is_categorical(series, threshold=0.3):
        unique_ratio = series.nunique() / len(series)
        return unique_ratio < threshold  # If unique values are less than threshold, treat as categorical

    # Iteratively Fill Missing Values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:  # Numeric columns
            df[col].fillna(df[col].mean(), inplace=True)  # Fill with mean
        elif df[col].dtype == 'object':  # String columns
            if is_categorical(df[col]):  # Check if categorical
                df[col].fillna(df[col].mode()[0], inplace=True)  # Fill with mode
            else:
                df[col].fillna("Unknown", inplace=True)  # Keep non-categorical data as "Unknown"

    # Encode categorical columns

    # print("\nData After Handling Missing Values:\n", df)
    return df


def clean_dataset(df, method="mean"):
    # Load dataset

    # Identify intentional duplicates
    intentional_duplicates = df.duplicated(keep='first')

    # Process each column based on its data type
    for column in df.columns:
        if df[column].duplicated().any():
            if df[column].dtype in ["int64", "float64"]:
                # Replace duplicate numerical values with mean or median
                if method == "mean":
                    df[column] = df[column].mask(df[column].duplicated() & ~intentional_duplicates, df[column].mean())
                elif method == "median":
                    df[column] = df[column].mask(df[column].duplicated() & ~intentional_duplicates, df[column].median())
            elif df[column].dtype == "object":
                # Check if the column is categorical
                if df[column].nunique() / len(df[column]) < 0.2:  # Assume categorized if unique values < 20% of total
                    df[column] = df[column].mask(df[column].duplicated() & ~intentional_duplicates,
                                                 df[column].mode()[0])
                else:
                    df[column] = df[column].mask(df[column].duplicated() & ~intentional_duplicates, None)

    # Print the cleaned dataset
    print("Cleaned Dataset:")
    # print(df)
    return df


# Example usage
# Replace with your actual dataset file


def detect_outliers_rf(df, contamination=0.05, n_estimators=100, random_state=42):
    """
    Detect outliers in a dataset using a Random Forest model.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    contamination (float): The proportion of outliers to detect (default: 0.05).
    n_estimators (int): Number of trees in the Random Forest (default: 100).
    random_state (int): Random seed for reproducibility (default: 42).

    Returns:
    pd.DataFrame: DataFrame with an additional column 'is_outlier' (1 if outlier, 0 otherwise).
    """
    df = df.copy()

    # Ensure numeric data only
    df = df.select_dtypes(include=[np.number]).dropna()

    if df.shape[1] < 2:
        raise ValueError("DataFrame must have at least two numeric columns.")

    # Splitting dataset into features and target
    X = df.drop(columns=df.columns[-1])  # Use all but last column as features
    y = df[df.columns[-1]]  # Use the last column as the target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Predict and compute errors
    y_pred = model.predict(X_test)
    errors = np.abs(y_pred - y_test)

    # Determine threshold for outliers
    threshold = np.percentile(errors, 100 * (1 - contamination))

    # Compute errors for the full dataset
    full_predictions = model.predict(X)
    full_errors = np.abs(full_predictions - y)

    # Identify outliers
    df['is_outlier'] = (full_errors > threshold).astype(int)

    return df


def detect_outliers_iforest(df, contamination=0.05, n_estimators=100, random_state=42):
    """
    Detect outliers in a dataset using the Isolation Forest algorithm.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    contamination (float): The proportion of outliers to detect (default: 0.05).
    n_estimators (int): Number of trees in the Isolation Forest (default: 100).
    random_state (int): Random seed for reproducibility (default: 42).

    Returns:
    pd.DataFrame: DataFrame with an additional column 'is_outlier' (1 if outlier, 0 otherwise).
    """
    df = df.copy()
    af = df.copy()
    # Ensure numeric data only
    df = df.select_dtypes(include=[np.number]).dropna()

    if df.shape[1] < 1:
        raise ValueError("DataFrame must have at least one numeric column.")

    # Train Isolation Forest
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    model.fit(df)

    # Predict outliers (-1 indicates an outlier, 1 indicates an inlier)
    af['is_outlier'] = (model.predict(df) == -1).astype(int)
    return af


# detect_anomalies('archive (1)/dataSet_1.csv')


def Enhancer(dataSettype, dataFrame):
    data_list = ['Medical', 'Financial or Billing', 'Survey']
    if data_list.__contains__(dataSettype):
        if dataSettype == 'Medical':
            return Main_1(dataFrame)
    # elif dataSettype=='Financial or Billing':
    #     Main_2(dataFrame)
    # elif dataSettype=='Survey':
    #    Main_3(dataFrame)


def Main(inp, df):
    #df = pd.read_csv(file_path)
    finalfile=Enhancer(inp, df)
    return finalfile


def Main_1(df):
    a = clean_dataset(df, method="mean")
    b = Fill_missing_values(a)
    c = detect_outliers_iforest(b, contamination=0.05, n_estimators=100, random_state=42)
    print(c)
    
    return c

#Main('Medical', 'archive (1)/dataSet_1.csv')





app = FastAPI()

# Enable CORS (Allow frontend to communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (replace "*" with specific frontend URL in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy ML Model (Replace with actual model)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), category: str = Form(...)):
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        csv_string = contents.decode("utf-8")
        df = pd.read_csv(StringIO(csv_string))

        # Store category in variable 'dts'
        dts = category
        print(f"Selected Category: {dts}")  # Debugging log

        # Process using ML Model
        processed_df = Main(df, dts)

        # Convert DataFrame back to CSV
        output = StringIO()
        processed_df.to_csv(output, index=False)
        output.seek(0)

        # Return file as response with correct headers
        return StreamingResponse(
            output, 
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=processed_file.csv"}
        )
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
