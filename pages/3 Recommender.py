import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

st.title("🏡 Property Recommendations")

# 1. Cache the data loading to speed up the app
@st.cache_data
def load_data():
    flats = pd.read_csv('flats.csv', index_col=0)
    houses = pd.read_csv('houses.csv', index_col=0)
    return flats, houses

flats, houses = load_data()

property_type = st.radio("Select Property Type", ['House', 'Flat'])

st.write("---")
st.subheader("⚙️ Recommender Settings")

# 2. Add Sliders for User Control
col1, col2 = st.columns(2)
with col1:
    num_recs = st.slider("How many recommendations?", min_value=1, max_value=10, value=5)
with col2:
    weight_slider = st.slider("Weight: Specs (Left) vs. Amenities/Luxury (Right)", 0.0, 1.0, 0.4)

# Calculate dynamic weights based on the slider
numeric_weight = 1.0 - weight_slider
text_weight = weight_slider


# ==========================================
# FLATS RECOMMENDER
# ==========================================
if property_type == 'Flat':
    utility_cols = [
        "Elevators", "Service Elevators in Building", "Lobby in Building",
        "Swimming Pool", "Security Staff", "Facilities for Disabled", "IsPrimeLoc",
    ]
    ohe_col = ['Building Type', 'Elevator Capacity']
    
    flat_pool = pd.get_dummies(flats, columns=ohe_col)
    flat_pool["Price_Per_Marla"] = flat_pool["Price(Cr)"] / flat_pool["Area(Marla)"]
    flat_pool["Description_Cleaned"] = flat_pool["Description"].fillna("").str.lower()

    flat_utility_cols = [
        "Elevators", "Floor in Building", "Building Type_Ground/Flat", "Building Type_High Rise",
        "Building Type_Lower Levels", "Building Type_Mid Rise Core", "Building Type_Skyscarper", 
        "Elevator Capacity_Dual Setup", "Elevator Capacity_High-Capacity", "Elevator Capacity_Multiple Bank",
        "Elevator Capacity_Single/None", "Parking Spaces"
    ]

    flat_pool[flat_utility_cols] = flat_pool[flat_utility_cols].fillna(0).astype(float)
    flat_numeric_group_cols = ["Price_Per_Marla", "Area(Marla)", "Bedroom(s)", "Bath(s)"] + flat_utility_cols
    
    society_flat_numeric = flat_pool.groupby("Building")[flat_numeric_group_cols].mean()
    society_flat_text = flat_pool.groupby("Building")["Description_Cleaned"].apply(lambda x: " ".join(x)).to_frame()
    flat_society_profiles = society_flat_numeric.merge(society_flat_text, left_index=True, right_index=True)

    def recommend_similar_flat_societies(selected_location, top_n=2):
        selected_location = str(selected_location).strip()

        if selected_location not in flat_society_profiles.index:
            return pd.DataFrame()

        target_profile = flat_society_profiles.loc[[selected_location]]
        alternative_profiles = flat_society_profiles[flat_society_profiles.index != selected_location].copy()

        if alternative_profiles.empty:
            return pd.DataFrame()

        scaler = MinMaxScaler()
        config_cols = ["Price_Per_Marla", "Area(Marla)", "Bedroom(s)", "Bath(s)"] + flat_utility_cols

        scaled_alt = scaler.fit_transform(alternative_profiles[config_cols])
        target_df = pd.DataFrame(target_profile[config_cols].values, columns=config_cols)
        scaled_target = scaler.transform(target_df)

        sim_numeric = cosine_similarity(scaled_target, scaled_alt).flatten()

        flat_vocabulary = [
            "penthouse", "studio", "luxury apartment", "standby generator",
            "reception lobby", "cctv security", "dedicated parking",
            "fast elevators", "monthly rent", "brand new",
        ]

        tfidf = TfidfVectorizer(vocabulary=flat_vocabulary)
        tfidf_alt = tfidf.fit_transform(alternative_profiles["Description_Cleaned"])
        tfidf_target = tfidf.transform(target_profile["Description_Cleaned"])

        sim_text = cosine_similarity(tfidf_target, tfidf_alt).flatten()

        # 3. Apply Dynamic Weights Here
        final_scores = (sim_numeric * numeric_weight) + (sim_text * text_weight)

        alternative_profiles["similarity_score"] = final_scores
        sorted_recommendations = alternative_profiles.sort_values(by="similarity_score", ascending=False)
        
        top_building_names = sorted_recommendations.head(top_n).index
        ui_display_table = society_flat_numeric.loc[top_building_names].copy()
        ui_display_table["similarity_score"] = sorted_recommendations.loc[top_building_names, "similarity_score"]
        
        return ui_display_table[["Price_Per_Marla", "Area(Marla)", "similarity_score"]]

    selected_building = st.selectbox("Select the Location", flats['Building'].dropna().unique())
    df = recommend_similar_flat_societies(selected_location=selected_building, top_n=num_recs)

    if not df.empty:
        # Convert to percentage and round
        df["similarity_score"] = df["similarity_score"] * 100
        df["Price_Per_Marla"] = df["Price_Per_Marla"].round(2)
        df["Area(Marla)"] = df["Area(Marla)"].round(1)

        df = df.rename(columns={'Price_Per_Marla': "Avg Price per Marla", 'Area(Marla)': "Avg Area (Marlas)", "similarity_score": "Match Score"})
        df.rename_axis(index={'Building': 'Recommended Buildings'}, inplace=True)

        # 4. Display visually with progress columns
        st.dataframe(
            df,
            column_config={
                "Match Score": st.column_config.ProgressColumn(
                    "Match Score (%)",
                    help="How closely this matches your selected location.",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Avg Price per Marla": st.column_config.NumberColumn("Avg Price per Marla (Cr)", format="%.2f"),
                "Avg Area (Marlas)": st.column_config.NumberColumn("Avg Area (Marlas)", format="%.1f")
            },
            use_container_width=True
        )


# ==========================================
# HOUSES RECOMMENDER
# ==========================================
else:
    data = houses
    utility_cols = ["SolarInstalled", "WaterBore", "CornerHouse", "IsPrimeLoc"]

    property_pool = data.copy()
    property_pool["Price_Per_Marla"] = property_pool["Price(Cr)"] / property_pool["Area(Marla)"]
    property_pool["Description_Cleaned"] = property_pool["Description"].fillna("").str.lower()
    property_pool[utility_cols] = property_pool[utility_cols].fillna(0).astype(int)

    numeric_group_cols = ["Price_Per_Marla", "Area(Marla)", "Bedroom(s)", "Bath(s)"] + utility_cols
    society_numeric_profiles = property_pool.groupby("Main Location")[numeric_group_cols].mean()
    society_text_profiles = property_pool.groupby("Main Location")["Description_Cleaned"].apply(lambda x: " ".join(x)).to_frame()
    
    society_profiles = society_numeric_profiles.merge(society_text_profiles, left_index=True, right_index=True)

    def recommend_similar_societies(selected_location, top_n=2):
        selected_location = str(selected_location).strip()

        if selected_location not in society_profiles.index:
            return pd.DataFrame()

        target_society = society_profiles.loc[[selected_location]]
        alternative_societies = society_profiles[society_profiles.index != selected_location].copy()

        scaler = MinMaxScaler()
        config_cols = ["Price_Per_Marla", "Area(Marla)", "Bedroom(s)", "Bath(s)", "SolarInstalled", "WaterBore"]

        scaled_alt_societies = scaler.fit_transform(alternative_societies[config_cols])
        target_df = pd.DataFrame(target_society[config_cols].values, columns=config_cols)
        scaled_target_society = scaler.transform(target_df)

        sim_numeric = cosine_similarity(scaled_target_society, scaled_alt_societies).flatten()

        luxury_vocabulary = [
            "double unit", "single unit", "brand new", "designer", "park face", 
            "open view", "gated community", "imported fittings", "luxury"
        ]

        tfidf = TfidfVectorizer(vocabulary=luxury_vocabulary)
        tfidf_alt = tfidf.fit_transform(alternative_societies["Description_Cleaned"])
        tfidf_target = tfidf.transform(target_society["Description_Cleaned"])

        sim_text = cosine_similarity(tfidf_target, tfidf_alt).flatten()

        # 3. Apply Dynamic Weights Here
        final_scores = (sim_numeric * numeric_weight) + (sim_text * text_weight)

        alternative_societies["similarity_score"] = final_scores
        sorted_recommendations = alternative_societies.sort_values(by="similarity_score", ascending=False)

        top_location_names = sorted_recommendations.head(top_n).index
        ui_display_table = society_numeric_profiles.loc[top_location_names].copy()
        ui_display_table["similarity_score"] = sorted_recommendations.loc[top_location_names, "similarity_score"]

        return ui_display_table[["Price_Per_Marla", "Area(Marla)", "similarity_score"]]

    selected_location = st.selectbox("Select the Location", data["Main Location"].dropna().unique())
    output_df = recommend_similar_societies(selected_location=selected_location, top_n=num_recs)

    if not output_df.empty:
        # Convert to percentage and round
        output_df["similarity_score"] = output_df["similarity_score"] * 100
        output_df["Price_Per_Marla"] = output_df["Price_Per_Marla"].round(2)
        output_df["Area(Marla)"] = output_df["Area(Marla)"].round(1)

        output_df = output_df.rename(
            columns={
                "Price_Per_Marla": "Avg Price per Marla",
                "Area(Marla)": "Avg Area (Marlas)",
                "similarity_score": "Match Score",
            }
        )
        output_df.index.name = "Recommended Location"

        # 4. Display visually with progress columns
        st.dataframe(
            output_df,
            column_config={
                "Match Score": st.column_config.ProgressColumn(
                    "Match Score (%)",
                    help="How closely this matches your selected location.",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Avg Price per Marla": st.column_config.NumberColumn("Avg Price per Marla (Cr)", format="%.2f"),
                "Avg Area (Marlas)": st.column_config.NumberColumn("Avg Area (Marlas)", format="%.1f")
            },
            use_container_width=True
        )