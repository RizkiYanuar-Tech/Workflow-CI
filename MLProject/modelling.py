import pandas as pd
import dagshub
import os
import mlflow
import argparse
import joblib
from dotenv import load_dotenv
from sklearn.ensemble  import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

# Load env key
load_dotenv()

# Load dan splitting data
def load_splitting_data(path):
    """
    Membaca dataset dari file CSV, memisahkan fitur dan target (Churn), 
    serta melakukan proporsi pembagian data untuk pelatihan dan pengujian.

    Args:
        filepath (str): Jalur atau lokasi file dataset CSV (contoh: 'preprocessing/clean_data.csv').

    Returns:
        tuple: Mengembalikan 4 buah variabel data dalam bentuk tuple:
            - X_train (DataFrame): Fitur yang digunakan untuk melatih mesin.
            - X_test (DataFrame): Fitur yang digunakan untuk menguji mesin.
            - y_train (Series): Kunci jawaban (target) untuk proses pelatihan.
            - y_test (Series): Kunci jawaban (target) untuk proses pengujian.
    """
    dataset = pd.read_csv(path)

    X = dataset.drop(columns=['Churn'])
    y = dataset['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

# Load Model
def load_train_model(feature_train, feature_test, label_train, label_test, args):
    """
    Melatih algoritma Random Forest Classifier dan secara otomatis mengirimkan 
    laporan performa (metrik) beserta file model akhir ke server DagsHub melalui MLflow.

    Args:
        X_train (DataFrame): Data fitur pelatihan.
        X_test (DataFrame): Data fitur pengujian.
        y_train (Series): Target/label data pelatihan.
        y_test (Series): Target/label data pengujian.

    Returns:
        None: Fungsi ini tidak mengembalikan nilai di dalam script, melainkan 
              langsung menyimpan artefak dan metrik ke cloud (DagsHub).
    """

    if "MLFLOW_RUN_ID" in os.environ:
        del os.environ["MLFLOW_RUN_ID"]

    dagshub.init(repo_owner='RizkiYanuar-Tech', repo_name='Workflow-CI', mlflow=True)

    mlflow.set_experiment("Base Model RandomForest")
    with mlflow.start_run(run_name='RF_Best_Model'):
        rf = RandomForestClassifier(random_state=42,
                                    class_weight='balanced',
                                    criterion=args.criterion,
                                    max_depth=args.max_depth,
                                    min_samples_split=args.min_samples_split,
                                    n_estimators=args.n_estimators
                                    )

        # Train
        rf.fit(feature_train, label_train)
        # Test
        y_pred = rf.predict(feature_test)

        accuracy = accuracy_score(label_test, y_pred)
        precision = precision_score(label_test, y_pred)
        recall = recall_score(label_test, y_pred)
        f1score = f1_score(label_test, y_pred)

        mlflow.log_metric('Accuracy', accuracy)
        mlflow.log_metric('Precision', precision)
        mlflow.log_metric('Recall', recall)
        mlflow.log_metric('f1_score', f1score)

        joblib.dump(rf, 'model.pkl')

        mlflow.sklearn.log_model(rf,
                                 'Random_Forest_Baseline',
                                 registered_model_name="Model_Churn")

        print('Model selesai dilatih dan disimpan dalam dagshub')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script Pelatihan Model RF Churn")
    parser.add_argument("--dataset", type=str, required=True, help='dataset')
    parser.add_argument("--criterion", type=str, default="gini")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=70)

    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_splitting_data(args.dataset)
    load_train_model(X_train, X_test, y_train, y_test, args)
