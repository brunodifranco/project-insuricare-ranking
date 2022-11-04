import pickle
import os
import pandas as pd
from flask import Flask, request, Response
from insuricare.insuricare import Insuricare

model = pickle.load(open('model/model_insuricare.pkl', 'rb'))

# initialize API
app = Flask(__name__)

# create endpoint
@app.route('/predict', methods=['POST'])
def health_insurance_predict():
    test_json = request.get_json()

    if test_json: # There is data
        if isinstance(test_json, dict): # Unique example 
            test_raw = pd.DataFrame(test_json, index = [0])
        else: # Multiple examples
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys()) 
        
        original_test_raw = test_raw.copy()

        # Instantiate Insuricare()
        pipeline = Insuricare()

        # Data Cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # Feature Engineering
        df2 = pipeline.feature_engineering(df1)

        # Data Preparation
        df3 = pipeline.data_preparation(df2)

        # Prediction
        df_response = pipeline.get_prediction(model, original_test_raw, df3)
        
        return df_response
       
    else: 
        return Response('{}', status = 200, mimetype = 'application/json')

if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000 )
    app.run( host = '0.0.0.0', port=port )