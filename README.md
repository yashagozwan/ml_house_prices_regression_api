## Langkah 8: Deployment and Monitoring

Pada langkah sebelumnya kita telah memiliki sebuah model regresi yang disimpan menjadi dua ekstensi yang berbeda yaitu joblib dan pickle.

Untuk google colab bisa di lihat di url ini [COLAB](https://colab.research.google.com/drive/19uIJSxtsjMUvXC2lX4ycwdorvDLhpcRX?usp=sharing)

### 8.1 Langkah pertama melakukan deployment adalah memuat model yang sudah kita bangun

```python
# Memuat model dari file joblib
joblib_model = joblib.load('models/gbr_model.joblib')

# Memuat model dari file pickle
with open('models/gbr_model.pkl', 'rb') as file:
    pickle_model = pickle.load(file)
```

### 8.2 Membangun Web Services sederhana

```python
from flask import Flask, request, jsonify
import joblib
import pickle

app = Flask(__name__)

# Memuat model dari file joblib
joblib_model = joblib.load('models/gbr_model.joblib')

# Memuat model dari file pickle
with open('models/gbr_model.pkl', 'rb') as file:
    pickle_model = pickle.load(file)



@app.route('/predict', methods=['POST'])
def predict():
    # Mengambil data dari request JSON
    data = request.json['data']
    # Melakukan prediksi (harus dalam bentuk 2D array)
    prediction = joblib_model.predict(data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
```
