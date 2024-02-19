import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

# Import ridge regressor and standard scaler pickle.
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

# Route for homepage

@app.route('/', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))
        except TypeError:
            return render_template('/home.html', result="Enter the form again")
        
        # Now we scale our data
        new_scaled_data = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        
        # Now we use our ridge pickle file to predict the result from the above entries.
        
        result = ridge_model.predict(new_scaled_data)
        
        return render_template('home.html', result=result[0])
    else:
        return render_template('home.html')
    
    
    
if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0", port=8000)