import pickle
import pandas as pd

class Insuricare():
    def __init__(self):
        self.path = ''
        self.age_scaler                  = pickle.load(open(self.path + 'scalers/age_scaler.pkl', 'rb'))
        self.annual_premium_scaler       = pickle.load(open(self.path + 'scalers/annual_premium_scaler.pkl', 'rb'))
        self.policy_sales_channel_scaler = pickle.load(open(self.path + 'scalers/policy_sales_channel_scaler.pkl', 'rb'))
        self.region_code_scaler          = pickle.load(open(self.path + 'scalers/region_code_scaler.pkl', 'rb'))
        self.vehicle_age_mms_scaler      = pickle.load(open(self.path + 'scalers/vehicle_age_mms_scaler.pkl', 'rb'))
        self.vehicle_age_oe_scaler       = pickle.load(open(self.path + 'scalers/vehicle_age_oe_scaler.pkl', 'rb'))
        self.vintage_scaler              = pickle.load(open(self.path + 'scalers/vintage_scaler.pkl', 'rb'))

    def data_cleaning(self, df1):
        df1['region_code'] = df1['region_code'].astype(str)
        df1['policy_sales_channel'] = df1['policy_sales_channel'].astype(str)
        return df1 

    def feature_engineering (self, df2):
        # vehicle age
        df2['vehicle_age'] = df2['vehicle_age'].apply(lambda x: 'below_1_year' if x=='< 1 Year'
                                                      else 'between_1_2_years' if x=='1-2 Year'
                                                      else 'above_2_years')
        # vehicle damage
        df2['vehicle_damage'] = df2['vehicle_damage'].apply(lambda x: 0 if x=='No' else 1)
        return df2

    def data_preparation (self, df3):
        # age
        df3['age'] = self.age_scaler.transform(df3[['age']].values)

        # vintage
        df3['vintage'] = self.vintage_scaler.transform(df3[['vintage']].values)

        # annual_premium
        df3['annual_premium'] = self.annual_premium_scaler.transform(df3[['annual_premium']].values)
        
        # gender - wasn't selected
        # df3 = pd.get_dummies(df3, prefix=['gender'], columns=['gender']).rename(columns={'gender_Female': 'female', 'gender_Male': 'male'})

        # region_code
        df3 = df3.assign(region_code=df3['region_code'].map(self.region_code_scaler))

        # policy_sales_channel
        df3 = df3.assign(policy_sales_channel=df3['policy_sales_channel'].map(self.policy_sales_channel_scaler))

        # vehicle_age - wasn't selected 
        # df3['vehicle_age'] = self.vehicle_age_oe_scaler.transform(df3[['vehicle_age']].values)
        # df3['vehicle_age'] = self.vehicle_age_mms_scaler.transform(df3[['vehicle_age']].values)

        # Feature Selection
        cols_selected = ['vintage', 'annual_premium', 'age', 'region_code', 'policy_sales_channel', 'vehicle_damage', 'previously_insured']

        return df3[cols_selected]


    def get_prediction(self, model, original_data, test_data):

        # Prediction
        pred = model.predict_proba(test_data)

        # Prediction as a column in the original data
        original_data['score'] = pred[:, 1].tolist()
        original_data = original_data.sort_values('score', ascending=False)

        return original_data.to_json(orient= 'records', date_format = 'iso')