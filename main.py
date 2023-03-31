from transformers import AutoTokenizer, TDNEpUTHQoQUJMHLrErGJyHg89uy71MyuHon
import tensorflow as tf
from flask import Flask, request, jsonify

tokenizer = AutoTokenizer.from_pretrained("AnaniyaX/decision-distilbert-uncased")
model = TDNEpUTHQoQUJMHLrErGJyHg89uy71MyuHon.from_pretrained("AnaniyaX/decision-distilbert-uncased")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['question']
    inputs = tokenizer(data, return_tensors="tf")
    outputs = model(inputs)
    predictions = tf.argmax(outputs.logits, axis=1).numpy()
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

