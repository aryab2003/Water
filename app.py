from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

app = Flask(__name__)


# Function to load data
def load_data():
    return pd.read_csv("water_potability.csv")


# Function to preprocess data
def preprocess_data(data):
    # Separate features and target
    X = data.drop("Potability", axis=1)
    y = data["Potability"]

    # Plan: IQR --> remove outliers, imputation, scaling
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    X = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
    y = y[X.index]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline for imputation and scaling
    pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )

    # Fit and transform the training data
    X_train_processed = pd.DataFrame(
        pipeline.fit_transform(X_train), columns=X_train.columns
    )

    # Transform the test data
    X_test_processed = pd.DataFrame(pipeline.transform(X_test), columns=X_test.columns)

    return X_train_processed, X_test_processed, y_train, y_test, pipeline


# Function to train models
def train_models(X_train, X_test, y_train, y_test):
    # Models to be trained
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, criterion="entropy", max_depth=100, random_state=42
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Classifier": CalibratedClassifierCV(
            SVC(kernel="linear", random_state=42), method="sigmoid"
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(
            max_depth=10,
            learning_rate=0.1,
            n_estimators=100,
            objective="binary:logistic",
            random_state=42,
        ),
    }

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {"model": model, "accuracy": accuracy}

    # Ensemble Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[(name, model["model"]) for name, model in results.items()],
        voting="soft",
    )
    voting_clf.fit(X_train, y_train)
    y_pred_voting = voting_clf.predict(X_test)
    accuracy_voting = accuracy_score(y_test, y_pred_voting)
    results["Voting Classifier"] = {"model": voting_clf, "accuracy": accuracy_voting}

    return results


# Load and preprocess data
data = load_data()
X_train, X_test, y_train, y_test, pipeline = preprocess_data(data)

# Train models
results = train_models(X_train, X_test, y_train, y_test)

# Save the best model
best_model_name = max(results, key=lambda x: results[x]["accuracy"])
best_model = results[best_model_name]["model"]
joblib.dump(best_model, "best_model.pkl")
joblib.dump(pipeline, "pipeline.pkl")


# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Load the model and pipeline
    model = joblib.load("best_model.pkl")
    pipeline = joblib.load("pipeline.pkl")

    # Get form data
    features = [float(x) for x in request.form.values()]
    user_data = pd.DataFrame(
        [features],
        columns=[
            "ph",
            "Hardness",
            "Solids",
            "Chloramines",
            "Sulfate",
            "Conductivity",
            "Organic_carbon",
            "Trihalomethanes",
            "Turbidity",
        ],
    )

    # Preprocess user input data
    user_data_processed = pd.DataFrame(
        pipeline.transform(user_data), columns=user_data.columns
    )

    # Predict
    prediction = model.predict(user_data_processed)
    result = "Potable" if prediction[0] == 1 else "Not Potable"

    return render_template("index.html", prediction_text=f"Prediction: {result}")


if __name__ == "__main__":
    app.run(debug=True)
