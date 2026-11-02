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