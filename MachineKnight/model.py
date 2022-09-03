import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler,MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

@st.cache(allow_output_mutation=True)  
def get_data_by_state():
	return pd.read_csv('train.csv')

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Housing Rent Prediction')
with dataset:
    data = get_data_by_state()
    le = LabelEncoder()
    label = le.fit_transform(data['id'])
    data.drop('id', axis=1, inplace=True)
    data["id"] = label
    data = data.drop(['id'],axis=1)
    label = le.fit_transform(data['activation_date'])
    data.drop('activation_date', axis=1, inplace=True)
    data["activation_date"] = label
    data = data.drop(['activation_date'],axis=1)
    label = le.fit_transform(data['amenities'])
    data.drop('amenities', axis=1, inplace=True)
    data["amenities"] = label
    data = data.drop(['amenities'],axis=1)
    label = le.fit_transform(data['locality'])
    data.drop('locality', axis=1, inplace=True)
    data["locality"] = label
    data = data.drop(['locality'],axis=1)
    label = le.fit_transform(data['longitude'])
    data.drop('longitude', axis=1, inplace=True)
    data["longitude"] = label
    data = data.drop(['longitude'],axis=1)
    label = le.fit_transform(data['latitude'])
    data.drop('latitude', axis=1, inplace=True)
    data["latitude"] = label
    data = data.drop(['latitude'],axis=1)
    label = le.fit_transform(data['negotiable'])
    data.drop('negotiable', axis=1, inplace=True)
    data["negotiable"] = label
    data = data.drop(['negotiable'],axis=1)
    label = le.fit_transform(data['property_age'])
    data.drop('property_age', axis=1, inplace=True)
    data["property_age"] = label
    data = data.drop(['property_age'],axis=1)
    label = le.fit_transform(data['cup_board'])
    data.drop('cup_board', axis=1, inplace=True)
    data["cup_board"] = label
    data = data.drop(['cup_board'],axis=1)
    data['type']=data['type'].map({'BHK1':1,'BHK2':2,'BHK3':3})
    data['furnishing']=data['furnishing'].map({'NOT_FURNISHED':0,'SEMI_FURNISHED':1,'FULLY_FURNISHED':2})
    data['parking']=data['parking'].map({'NONE':0,'TWO_WHEELER':2,'FOUR_WHEELER':4,'BOTH':1})
    data['lease_type']=data['lease_type'].map({'FAMILY':0,'ANYONE':1,'BACHELOR':2,'COMPANY':3})
    data['facing']=data['facing'].map({'N':0,'NE':1,'NW':2,'E':3,'SE':4,'S':5,'SW':6,'W':7})
    data['water_supply']=data['water_supply'].map({'CORP_BORE':0,'CORPORATION':1,'BOREWELL':2})
    data['building_type']=data['building_type'].map({'AP':0,'IF':1,'IH':2,'GC':3})
    target = np.array(data.drop(['rent'],1))
    features = np.array(data['rent'])
    
    
    
with model_training:
    sel_col,disp_col = st.columns(2)

    
    
    x_train , x_test , y_train , y_test = train_test_split(target,features,test_size=0.25,random_state=42)
    gb = HistGradientBoostingRegressor
    regr = gb(max_depth=100)
    regr.fit(x_train,y_train)
    y_pred = regr.predict(x_test)
    
    def gb_param_selector(type_of_house,lease_type,gym,lift,swimming_pool,furnishing,parking,property_size,bathroom,facing,floor,total_floor,water_supply,building_type,balconies):
        prediction = regr.predict([[type_of_house,lease_type,gym,lift,swimming_pool,furnishing,parking,property_size,bathroom,facing,floor,total_floor,water_supply,building_type,balconies]])
        return prediction


    
           
    
    type_of_house = st.number_input('Enter the type of house(1 for 1BHK, 2 for 2BHK, 3 for 3BHK)',value=1)
    lease_type = st.number_input('Enter the lease type(0 for Family, 1 for anyone, 2 for bachelor, 3 for company)',value=1) 
    gym = st.number_input('Is gym available?(0 for no, 1 for yes)',value=1) 
    lift = st.number_input('Is lift available?(0 for no, 1 for yes)',value=1)
    swimming_pool = st.number_input('Is swimming pool available(0 for no, 1 for yes)',value=1)
    furnishing = st.number_input('Level of furnishing in the house(0 for not furnished, 1 for semi, 2 for fully furnished)',value=1)
    parking = st.number_input('Type of parking provided(0 for none, 2 for two-wheeler,4 for four-wheeler,1 for both)',value=1)
    property_size = st.number_input('Size of the property',value=1)
    bathroom = st.number_input('Number of bathrooms',value=1)
    facing = st.number_input('Direction of the house is(Starts from 0 in clockwise direction from north)',value=1)
    floor = st.number_input('Floor number of the house',value=1)
    total_floor = st.number_input('Total floors in the building',value=1)
    water_supply = st.number_input('Water supply is done from(0 for corp bore, 1 for corporation, 2 for borewell)',value=1)
    building_type = st.number_input('Type of the building(0 for AP, 1 for IF, 2 for IH, 3 for GC)',value=1)
    balconies = st.number_input('Number of balconies',value=1)
    result =""
      
    
    if st.button("Predict"): 
        result = gb_param_selector(type_of_house,lease_type,gym,lift,swimming_pool,furnishing,parking,property_size,bathroom,facing,floor,total_floor,water_supply,building_type,balconies) 
        st.success('The probable value for rent of the house is {}'.format(result))
         
    
