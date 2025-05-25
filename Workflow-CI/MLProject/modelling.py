import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import argparse
import joblib

def main(data_path):
    mlflow.set_experiment("pollution-classifier")  

    # Aktifkan autolog untuk rekam otomatis parameter, metrik, model
    mlflow.sklearn.autolog()

    # Load dataset
    df = pd.read_csv(data_path)
    X = df.drop(columns="Air Quality")
    y = df["Air Quality"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        # Training model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Simpan model secara manual sebagai file .pkl di folder kerja
        joblib.dump(model, "model.pkl")

        # Log artefak model.pkl ke MLflow tracking server
        mlflow.log_artifact("model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset CSV")
    args = parser.parse_args()
    main(args.data_path)
