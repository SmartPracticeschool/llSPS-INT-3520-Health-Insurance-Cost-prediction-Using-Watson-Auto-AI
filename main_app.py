from flask import Flask,request, url_for, redirect, render_template
import pickle 
import numpy as np

app = Flask(__name__)

model = pickle.load(open('insurance_model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("health_insurance.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    features = [float(x) for x in request.form.values()]
    final = [np.array(features)]
    print(features)
    print(final)
    prediction = model.predict(final)
    output = '{0:.{1}f}'.format(prediction[0], 3)

    return render_template('health_insurance.html',pred='The Predicted Insurance Amount base on your input is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
