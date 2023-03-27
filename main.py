from flask import Flask

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello from Flask!'


app.run(host='0.0.0.0', port=81)
from transformers import AutoTokenizer, TFDistilBertModel
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("AnaniyaX/distilbert-base-uncased")
model = TFDistilBertModel.from_pretrained("AnaniyaX/distilbert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
outputs = model(inputs)
n = tokenizer(["who is down there"], truncation=True, padding=True)
# n = tf.data.Dataset.from_tensor_slices(dict(n)).batch(16)
print(n)

predictions = model.predict(n)
print(predictions)
predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()

print(predicted_labels)