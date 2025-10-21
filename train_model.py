import mlflow
from mlflow.models import infer_signature

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("http://localhost:5001") 

mlflow.set_experiment("iris-classification")

def train_model(n_estimators=100, max_depth=5, random_state=42):
    """
    Train a Random Forest classifier on Iris dataset
    """

    with mlflow.start_run():
        
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=random_state
        )
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Infer and log model signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Log model
        model_info = mlflow.sklearn.log_model(
            sk_model=model, 
            name="iris_rf_model",
            signature=signature,
            input_example=X_train[:5],
            registered_model_name="IrisRandomForestModel"
        )
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return model

if __name__ == "__main__":
    train_model(n_estimators=1, max_depth=1)