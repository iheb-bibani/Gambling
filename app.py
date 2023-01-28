import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('finalized_model.sav')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    
    if output == 0:
        text = "It's Green , Play !"
    else : 
        text = "It's not Green , Don't Play!"

    return render_template('index.html', prediction_text='Next Color Should be {}'.format(text))


if __name__ == "__main__":
    app.run(debug=True)
