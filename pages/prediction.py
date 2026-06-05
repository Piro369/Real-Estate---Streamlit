import numpy as np
import pandas as pd
import streamlit as st
import joblib
import __main__


X_train = pd.read_csv('Flats/X_train.csv',index_col=0).drop(columns=['Price per Unit','Elevators','Floor in Building','Floor','Store Rooms','luxury_type','Building Type'])
X_test = pd.read_csv('Flats/X_test.csv',index_col=0).drop(columns=['Price per Unit','Elevators','Floor in Building','Floor','Store Rooms','luxury_type','Building Type'])
y_train = pd.read_csv('Flats/y_train.csv',index_col=0)
y_test = pd.read_csv('Flats/y_test.csv',index_col=0)

X_train_houses = pd.read_csv('Houses/X_train.csv',index_col=0)
X_test_houses = pd.read_csv('Houses/X_test.csv',index_col=0)
y_train_houses = pd.read_csv('Houses/y_train.csv',index_col=0)
y_test_houses = pd.read_csv('Houses/y_test.csv',index_col=0)




from sklearn.base import BaseEstimator,TransformerMixin

class CustomOrdinalMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mappings):
        self.mappings = mappings
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Handle case if X is a NumPy array, convert to DataFrame for column mapping
        if not isinstance(X, pd.DataFrame):
            X_copy = pd.DataFrame(X)
        else:
            X_copy = X.copy()
            
        for col, mapping_dict in self.mappings.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(mapping_dict).fillna(1).astype(int)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Returns the input features as the output features."""
        if input_features is None:
            return np.array(list(self.mappings.keys()))
        return np.asarray(input_features)
    
__main__.CustomOrdinalMapper = CustomOrdinalMapper
    
all_mappings = {
    'Property era': {
        'Vintage': 1, 'Established': 2, 'Recently Built': 3, 
        'Modern Era': 4, 'Brand New (2025)': 5, 'Future': 6
    },
    'Floor Level': {
        'Lower Floors': 1, 'Ground/First Floor': 2, 'Mid-Level': 3, 'High-Rise': 4
    },
    'Elevator Capacity': {
        'Single/None': 1, 'Dual Setup': 2, 'Multiple Bank': 3, 'High-Capacity': 4
    }
}

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        """Custom Transformer for Log Transformation."""
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = np.copy(X)
        return np.log1p(X_copy)
        
    def inverse_transform(self, X):
        X_copy = np.copy(X)
        return np.expm1(X_copy) 

    def get_feature_names_out(self, input_features=None):
        """Returns the input features as the output features."""
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out.")
        return np.asarray(input_features)

__main__.LogTransformer = LogTransformer

class CustomOrdinalMapperHouses(BaseEstimator, TransformerMixin):
    def __init__(self, mappings):
        self.mappings = mappings
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Handle case if X is a NumPy array, convert to DataFrame for column mapping
        if not isinstance(X, pd.DataFrame):
            X_copy = pd.DataFrame(X)
        else:
            X_copy = X.copy()
            
        for col, mapping_dict in self.mappings.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(mapping_dict).fillna(1).astype(int)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Returns the input features as the output features."""
        if input_features is None:
            return np.array(list(self.mappings.keys()))
        return np.asarray(input_features)
    
__main__.CustomOrdinalMapperHouses = CustomOrdinalMapperHouses
    
all_mappings_houses = {
    'Property era': {
        'Vintage': 1, 'Established': 2, 'Recently Built': 3, 
        'Modern Era': 4, 'Brand New (2025)': 5, 'Future': 6
    }
}


# //////////////////////////////////////////////////////////////////


st.title("Property Price Prediction")

property_type = st.radio("Select Property Type",['House','Flat'])

if property_type == 'Flat':
    st.subheader("Predict Flat Prices")

    try:
        pipeline = joblib.load("pipeline_pickle_flats.joblib")

        main_location = st.selectbox("Main Location", X_train['Main Location'].value_counts().index)

        valid_locations = X_train[X_train['Main Location'] == main_location]['Building'].unique()
        building = st.selectbox("Building", valid_locations)

        area = st.number_input("Area (Marla)", min_value=1.0, max_value=30.0,step=0.1)
        beds = st.number_input("Bedrooms", min_value=X_train['Bedroom(s)'].min(), max_value=X_train['Bedroom(s)'].max(),step=1.0)
        baths = st.number_input("Bathrooms",min_value=X_train['Bath(s)'].min(), max_value=X_train['Bath(s)'].max(), step=1.0)

        parking_spaces = st.number_input("Parking Spaces", min_value=1, max_value=4)
        kitchens = st.number_input("Kitchens", min_value=1, max_value=3)
        servant_quarters = st.number_input("Servant Quarters", min_value=0, max_value=3)

        property_era = st.selectbox("Property Era", X_train['Property era'].dropna().unique())
        floor_level = st.selectbox("Floor Level", X_train['Floor Level'].dropna().unique())
        elevator_capacity = st.selectbox("Elevator Capacity", X_train['Elevator Capacity'].dropna().unique())

       
        if st.button("Predict"):

            input_dict = {
                "Main Location": [main_location],
                "Building": [building],
                "Area(Marla)": [area], 
                "Bedroom(s)": [beds],
                "Bath(s)": [baths],
                "Parking Spaces": [parking_spaces],
                "Kitchens": [kitchens],
                "Servant Quarters": [servant_quarters],
                "Property era": [property_era],
                "Floor Level": [floor_level],
                "Elevator Capacity": [elevator_capacity]
            }
            
            property_features = pd.DataFrame(input_dict)
            
            prediction = np.expm1(pipeline.predict(property_features))
            
            st.success(f"Estimated Price: Rs. {prediction[0]:,.2f}  Crore")

    except FileNotFoundError:
        st.error("Model File not Found")

elif property_type == "House":
    st.subheader("Predict House Prices")

    try:
        pipeline = joblib.load("pipeline_pickle_houses.joblib")


        main_location = st.selectbox("Main Location", X_train_houses['Main Location'].value_counts().index)

        area = st.number_input("Area (Marla)", min_value=1.0, max_value=X_train_houses['Area(Marla)'].max(),step=0.1)
        beds = st.number_input("Bedrooms", min_value=X_train_houses['Bedroom(s)'].min(), max_value=X_train['Bedroom(s)'].max(),step=1.0)
        baths = st.number_input("Bathrooms",min_value=X_train_houses['Bath(s)'].min(), max_value=X_train['Bath(s)'].max(), step=1.0)

        kitchens = st.number_input("Kitchens", min_value=1, max_value=3)
        servant_quarters = st.number_input("Servant Quarters", min_value=0, max_value=3)

        property_era = st.selectbox("Property Era", X_train_houses['Property era'].dropna().unique())
        
        store_rooms = st.number_input("Store Rooms", min_value=0, max_value=3)

        storey_unit = st.number_input("Storey Units",min_value=1,max_value=3)

        prime_location = st.selectbox("Prime Location", X_train_houses['IsPrimeLoc'].value_counts().index)
        is_solar_installed = st.selectbox("Solar Installed", X_train_houses['SolarInstalled'].value_counts().index)
        water_bore = st.selectbox("Water Bore", X_train_houses['WaterBore'].value_counts().index)
        corner_house = st.selectbox("Corner House", X_train_houses['CornerHouse'].value_counts().index)

        luxury_type = st.selectbox("Luxury Type", ['Luxury','Budget'])

        luxury_type = 0 if luxury_type == 'Luxury' else 1
       
        if st.button("Predict"):

            input_dict = {
                "Main Location": [main_location],
                "Area(Marla)": [area], 
                "Bedroom(s)": [beds],
                "Bath(s)": [baths],
                "Kitchens": [kitchens],
                "Servant Quarters": [servant_quarters],
                "Property era": [property_era],
                "Store Rooms":[store_rooms],
                "Storey Unit":[storey_unit],
                "IsPrimeLoc":[prime_location],
                "SolarInstalled":[is_solar_installed],
                "WaterBore":[water_bore],
                "CornerHouse":[corner_house],
                "luxury_type":[luxury_type]


            }
            
            property_features = pd.DataFrame(input_dict)
            
            prediction = np.expm1(pipeline.predict(property_features))
            
            st.success(f"Estimated Price: Rs. {prediction[0]:,.2f}  Crore")

    except FileNotFoundError:
        st.error("Model File not Found")