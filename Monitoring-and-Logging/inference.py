import pandas as pd
import joblib
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
model = joblib.load('model.pkl')

# Metrik Prometheus
request_count = Counter('model_requests_total', 'Total permintaan')
request_latency = Histogram('model_request_latency_seconds', 'Latensi permintaan')
prediction_accuracy = Gauge('model_prediction_accuracy', 'Akurasi prediksi')
request_errors = Counter('model_request_errors_total', 'Total error')
cpu_usage = Gauge('model_cpu_usage_percent', 'Penggunaan CPU')
memory_usage = Gauge('model_memory_usage_percent', 'Penggunaan Memori')
precision_metric = Gauge('model_precision', 'Presisi')
recall_metric = Gauge('model_recall', 'Recall')
f1_metric = Gauge('model_f1_score', 'F1 Score')
response_time = Gauge('model_response_time_seconds', 'Waktu respons')

# List penyimpanan label dan prediksi untuk hitung metrik
y_true = []
y_pred = []

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    request_count.inc()
    try:
        data = request.get_json(force=True)

        # Pastikan data mengandung fitur dan label 'Air Quality' (atau sesuaikan)
        if "Air Quality" not in data:
            return jsonify({'error': 'Missing target label "Air Quality" in input data for evaluation'}), 400
        
        label = data.pop("Air Quality")
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]

        # Simpan hasil untuk hitung metrik evaluasi
        y_true.append(label)
        y_pred.append(prediction)

        # Hitung metrik evaluasi
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Update Prometheus metrics
        prediction_accuracy.set(accuracy)
        precision_metric.set(precision)
        recall_metric.set(recall)
        f1_metric.set(f1)
        cpu_usage.set(psutil.cpu_percent())
        memory_usage.set(psutil.virtual_memory().percent)
        response_time.set(time.time() - start_time)

        with request_latency.time():
            time.sleep(0.1)

        return jsonify({
            'prediction': int(prediction),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
    except Exception as e:
        request_errors.inc()
        return jsonify({'error': str(e)}), 400

@app.route('/features', methods=['GET'])
def get_features():
    try:
        features = list(model.feature_names_in_)
    except AttributeError:
        features = []
    return jsonify({"features": features})

if __name__ == '__main__':
    start_http_server(8001)
    app.run(host='0.0.0.0', port=8000)
