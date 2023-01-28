import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('finalized_model.sav')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
    


if __name__ == "__main__":
    app.run(debug=True)
