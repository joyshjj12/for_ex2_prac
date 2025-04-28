import joblib
from flask import request, jsonify, Flask
import numpy as np


app = Flask(__name__)
clf = joblib.load('model/model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    new_sample = np.array(data['sample']).reshape(1, -1)
    prediction = clf.predict(new_sample)
    class_names = ['Iris Setosca', 'Iris Virginica', 'Iris Versicolor']
    predicted_class = class_names[prediction[0]]
    return jsonify({'predicted class':predicted_class})


if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)