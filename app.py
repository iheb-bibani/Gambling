import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('finalized_model.sav')

@app.route('/')
def home():
    return render_template('index.html')

def predict():
    feature_list = request.form.to_dict()
    feature_list = list(feature_list.values())
    feature_list = list(map(int, feature_list))
    final_features = np.array(feature_list).reshape(1, 27)
    prediction = model.predict(final_features)

    output = prediction[0]

    if output == 0:
        text = "It's Green , Play !"
    else:
        text = "It's Not Green , Don't Play!"

    return render_template('index.html', prediction_text='{}'.format(text))
    
if __name__ == "__main__":
    app.run(debug=True)
