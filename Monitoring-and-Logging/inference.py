import pandas as pd
import joblib
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import psutil

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

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    request_count.inc()
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        accuracy, precision, recall, f1 = 0.95, 0.90, 0.92, 0.91  # Simulasi
        prediction_accuracy.set(accuracy)
        precision_metric.set(precision)
        recall_metric.set(recall)
        f1_metric.set(f1)
        cpu_usage.set(psutil.cpu_percent())
        memory_usage.set(psutil.virtual_memory().percent)
        response_time.set(time.time() - start_time)
        with request_latency.time():
            time.sleep(0.1)
        return jsonify({'prediction': int(prediction), 'accuracy': accuracy})
    except Exception as e:
        request_errors.inc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    start_http_server(8001)
    app.run(host='0.0.0.0', port=8000)