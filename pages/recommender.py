import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


st.title("Property Recommendations")
# Add your recommendation logic here later

flats = pd.read_csv('../Missing Values Imputation/flats.csv',index_col=0)
houses = pd.read_csv('../Missing Values Imputation/houses.csv',index_col=0)


property_type = st.radio("Select Property Type",['House','Flat'])

if property_type == 'Flat':

    utility_cols = [
    "Elevators",
    "Service Elevators in Building",
    "Lobby in Building",
    "Swimming Pool",
    "Security Staff",
    "Facilities for Disabled",
    "IsPrimeLoc",
    ]
    ohe_col = [
        'Building Type',
        'Elevator Capacity',
    ]
    flat_pool = pd.get_dummies(flats, columns=ohe_col)

    flat_pool["Price_Per_Marla"] = (
        flat_pool["Price(Cr)"] / flat_pool["Area(Marla)"]
    )

    flat_pool["Description_Cleaned"] = (
        flat_pool["Description"].fillna("").str.lower()
    )

    # Core structural, luxury, and encoded columns unique to vertical complexes
    flat_utility_cols = [
        "Elevators",
        "Floor in Building",
        "Building Type_Ground/Flat", "Building Type_High Rise",
        "Building Type_Lower Levels", "Building Type_Mid Rise Core",
        "Building Type_Skyscarper", "Elevator Capacity_Dual Setup",
        "Elevator Capacity_High-Capacity", "Elevator Capacity_Multiple Bank",
        "Elevator Capacity_Single/None",
        "Parking Spaces"
    ]

    # Convert features to numeric and handle missing values safely
    flat_pool[flat_utility_cols] = flat_pool[flat_utility_cols].fillna(0).astype(float)


    # --- STEP 2: COLLAPSE FLATS INTO DISTINCT SOCIETY PROFILES ---

    # Group numerical features to compute the average apartment standard per building cluster
    flat_numeric_group_cols = ["Price_Per_Marla", "Area(Marla)", "Bedroom(s)", "Bath(s)"] + flat_utility_cols
    society_flat_numeric = (
        flat_pool.groupby("Building")[flat_numeric_group_cols].mean()
    )

    # Combine all apartment descriptions within a building cluster into a single text block
    society_flat_text = (
        flat_pool.groupby("Building")["Description_Cleaned"]
        .apply(lambda x: " ".join(x))
        .to_frame()
    )

    # Merge numeric profiles and text blocks into a master Flat Society Profile
    flat_society_profiles = society_flat_numeric.merge(
        society_flat_text, left_index=True, right_index=True
    )


    # --- STEP 3: THE LOCATION-TO-LOCATION FLAT ENGINE ---
    def recommend_similar_flat_societies(selected_location, top_n=2):
        selected_location = str(selected_location).strip()

        if selected_location not in flat_society_profiles.index:
            return f"Location '{selected_location}' has no flat listings. Options: {list(flat_society_profiles.index)}"

        # Isolate the target society profile
        target_profile = flat_society_profiles.loc[[selected_location]]

        # Exclude the target society to ensure we recommend alternative locations
        alternative_profiles = flat_society_profiles[
            flat_society_profiles.index != selected_location
        ].copy()

        if alternative_profiles.empty:
            return "No alternative locations containing flat profiles were found."

        # PASS 1: Calculate Spatial, Structural, & Building Utility Similarity
        scaler = MinMaxScaler()
        
        # FIXED: Ensure your dummy features are actually included in the similarity calculation
        config_cols = [
            "Price_Per_Marla",
            "Area(Marla)",
            "Bedroom(s)",
            "Bath(s)",
        ] + flat_utility_cols

        scaled_alt = scaler.fit_transform(alternative_profiles[config_cols])
        
        # FIXED: Wrapped row cleanly to avoid dimensional scikit-learn warnings/errors
        target_df = pd.DataFrame(target_profile[config_cols].values, columns=config_cols)
        scaled_target = scaler.transform(target_df)

        sim_numeric = cosine_similarity(scaled_target, scaled_alt).flatten()

        # PASS 2: Extract High-Rise Luxury Vibe (TF-IDF Text Mining)
        flat_vocabulary = [
            "penthouse", "studio", "luxury apartment", "standby generator",
            "reception lobby", "cctv security", "dedicated parking",
            "fast elevators", "monthly rent", "brand new",
        ]

        tfidf = TfidfVectorizer(vocabulary=flat_vocabulary)
        tfidf_alt = tfidf.fit_transform(alternative_profiles["Description_Cleaned"])
        tfidf_target = tfidf.transform(target_profile["Description_Cleaned"])

        sim_text = cosine_similarity(tfidf_target, tfidf_alt).flatten()

        # PASS 3: Combined Weighted Score (Specs/Infrastructure: 60%, Text Vibe: 40%)
        final_scores = (sim_numeric * 0.60) + (sim_text * 0.40)

        # Attach scores and rank locations
        alternative_profiles["similarity_score"] = final_scores
        sorted_recommendations = alternative_profiles.sort_values(
            by="similarity_score", ascending=False
        )
        
        # Grab the top top_n indices (the names of the buildings)
        top_building_names = sorted_recommendations.head(top_n).index
        
        # Pull the ORIGINAL unscaled metrics from your base profile frame
        ui_display_table = society_flat_numeric.loc[top_building_names].copy()
        
        # Attach the calculated similarity scores cleanly
        ui_display_table["similarity_score"] = sorted_recommendations.loc[top_building_names, "similarity_score"]
        
        # Return this unscaled table to your Streamlit front-end
        return ui_display_table[
            [
                "Price_Per_Marla",
                "Area(Marla)",
                "similarity_score"
            ]
        ]


    # --- RUN TEST QUERY ---
    selected_building = st.selectbox("Select the Location",flats['Building'].unique())

    df = recommend_similar_flat_societies(selected_location=selected_building, top_n=5)

    df = df.rename(columns={'Price_Per_Marla':"Avg Price per Marla",'Area(Marla)':"Avg Area(Marlas)"})
    df.rename_axis(index={'Building':'Recommended Buildings'},inplace=True)

    st.dataframe(df)

    
# ////////////////////////////////////////////////////////////////////////////////
else:
    data = houses
    utility_cols = ["SolarInstalled", "WaterBore", "CornerHouse", "IsPrimeLoc"]

    # Load your dataset
    property_pool = data.copy()

    # --- STEP 1: CALCULATE BASE PROPERTY LEVEL METRICS ---
    property_pool["Price_Per_Marla"] = (
        property_pool["Price(Cr)"] / property_pool["Area(Marla)"]
    )
    property_pool["Description_Cleaned"] = (
        property_pool["Description"].fillna("").str.lower()
    )

    # Convert boolean columns to integers for mathematical averaging
    property_pool[utility_cols] = property_pool[utility_cols].fillna(0).astype(int)


    # --- STEP 2: COLLAPSE HOUSES INTO DISTINCT SOCIETY PROFILES ---

    # Group numeric metrics by Main Location (calculating the mean lifestyle profile)
    numeric_group_cols = [
        "Price_Per_Marla",
        "Area(Marla)",
        "Bedroom(s)",
        "Bath(s)",
    ] + utility_cols
    society_numeric_profiles = property_pool.groupby("Main Location")[
        numeric_group_cols
    ].mean()

    # Concatenate all descriptions in a society together to create a single text pool per location
    society_text_profiles = (
        property_pool.groupby("Main Location")["Description_Cleaned"]
        .apply(lambda x: " ".join(x))
        .to_frame()
    )

    # Merge numeric and text profiles into a master Society Profile DataFrame
    society_profiles = society_numeric_profiles.merge(
        society_text_profiles, left_index=True, right_index=True
    )


    # --- STEP 3: THE LOCATION-TO-LOCATION ENGINE ---
    def recommend_similar_societies(selected_location, top_n=2):
        # Standardize input string
        selected_location = str(selected_location).strip()

        # Isolate the target society profile row
        target_society = society_profiles.loc[[selected_location]]

        # Exclude the target society itself from the matching pool to discover alternatives
        alternative_societies = society_profiles[
            society_profiles.index != selected_location
        ].copy()

        # PASS 1: Calculate Numeric & Infrastructure Similarity
        scaler = MinMaxScaler()
        config_cols = [
            "Price_Per_Marla",
            "Area(Marla)",
            "Bedroom(s)",
            "Bath(s)",
            "SolarInstalled",
            "WaterBore",
        ]

        scaled_alt_societies = scaler.fit_transform(
            alternative_societies[config_cols]
        )

        # FIXED: Protect schema names during transform pass
        target_df = pd.DataFrame(
            target_society[config_cols].values, columns=config_cols
        )
        scaled_target_society = scaler.transform(target_df)

        sim_numeric = cosine_similarity(
            scaled_target_society, scaled_alt_societies
        ).flatten()

        # PASS 2: Calculate Lifestyle/Luxury Vibe Similarity via TF-IDF Text Mining
        luxury_vocabulary = [
            "double unit",
            "single unit",
            "brand new",
            "designer",
            "park face",
            "open view",
            "gated community",
            "imported fittings",
            "luxury",
        ]

        tfidf = TfidfVectorizer(vocabulary=luxury_vocabulary)
        tfidf_alt = tfidf.fit_transform(alternative_societies["Description_Cleaned"])
        tfidf_target = tfidf.transform(target_society["Description_Cleaned"])

        sim_text = cosine_similarity(tfidf_target, tfidf_alt).flatten()

        # PASS 3: Combine Scores (Numerical Specs: 60%, Text Luxury Vibe: 40%)
        final_scores = (sim_numeric * 0.60) + (sim_text * 0.40)

        # Attach scores and rank
        alternative_societies["similarity_score"] = final_scores
        sorted_recommendations = alternative_societies.sort_values(
            by="similarity_score", ascending=False
        )

        # Grab the top top_n indices (the names of the locations)
        top_location_names = sorted_recommendations.head(top_n).index

        # FIXED: Pull the ORIGINAL unscaled metrics from numeric profiles instead of text pool
        ui_display_table = society_numeric_profiles.loc[top_location_names].copy()

        # Attach the calculated similarity scores cleanly
        ui_display_table["similarity_score"] = sorted_recommendations.loc[
            top_location_names, "similarity_score"
        ]

        # Return this unscaled table to your Streamlit front-end
        return ui_display_table[["Price_Per_Marla", "Area(Marla)", "similarity_score"]]


    # --- STREAMLIT UI LAYOUT ---
    selected_location = st.selectbox(
        "Select the Location", data["Main Location"].unique()
    )

    # Run logic
    output_df = recommend_similar_societies(
        selected_location=selected_location, top_n=5
    )

    # Format column names and indices for presentation layout
    output_df = output_df.rename(
        columns={
            "Price_Per_Marla": "Avg Price per Marla",
            "Area(Marla)": "Avg Area (Marlas)",
            "similarity_score": "Match Score",
        }
    )
    output_df.index.name = "Recommended Location"

    st.dataframe(output_df)



