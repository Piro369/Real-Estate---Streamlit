import streamlit as st
import pandas as pd
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

flats = pd.read_csv('flats.csv',index_col=0)
houses = pd.read_csv('houses.csv',index_col=0)

flats_coord = pd.read_csv('flats_lat_long.csv',index_col=0)
houses_coord = pd.read_csv('houses_lat_long.csv',index_col=0)

st.title("Market Analysis")
property_type = st.radio("Select Property Type",['House','Flat'])

if property_type == 'Flat':
    data = flats
    data_coord = flats_coord
else:
    data = houses
    data_coord = houses_coord

st.subheader("Scatter Plot")

data_coord = data_coord.dropna(subset=['Latitude', 'Longitude', 'Price(Cr)']) # Drop empty rows
grouped_df = data_coord.groupby(['Main Location', 'Latitude', 'Longitude']).agg(
    Average_Price=('Price(Cr)', 'mean'),   # Calculates the mean price for this location
    Total_Listings=('Price(Cr)', 'count')  # Counts how many flats are in this location
).reset_index()

# Optional: Round the price to 2 decimal places so the tooltip looks clean
grouped_df['Average_Price'] = grouped_df['Average_Price'].round(2)

# 2. Build the Density Map
fig = px.scatter_mapbox(
    grouped_df,
    lat='Latitude',
    lon='Longitude',
    color='Average_Price',            # Color is now based on the calculated MEAN
    size='Total_Listings',            # Bubble size is based on how many flats are there
    color_continuous_scale="Plasma",
    size_max=30,
    center=dict(lat=33.6844, lon=73.0479),
    zoom=10.5,
    mapbox_style="carto-darkmatter",
    hover_name="Main Location",
    
    # Customize the hover tooltip to show exactly what we want
    hover_data={
        'Latitude': False, 
        'Longitude': False, 
        'Average_Price': True, 
        'Total_Listings': True
    },
    title="Average Property Price per Location in Crores"
)

fig.update_layout(margin={"r":0, "t":40, "l":0, "b":0})
st.plotly_chart(fig, use_container_width=True)



st.subheader("Word Cloud")


text = " ".join(desc for desc in data['Description'].dropna())

# 2. Fine-tune your stopwords
# We start with default English stopwords (the, and, a, etc.)
custom_stopwords = set(STOPWORDS)

# Real estate descriptions often need extra filtering to be useful.
# Add words here that you want to ignore because they are too generic.
real_estate_noise = {
    'sale', 'available', 'detail', 'project', 'option', 'best', 'prime', 
    'location', 'ideal', 'contact', 'us', 'please', 'today', 'make', 'one', 
    'view', 'brand', 'new', 'looking', 'find', 'feel', 'free', 'investment', 
    'investor', 'opportunity', 'ready', 'value', 'quality', 'high', 'type',
    'real', 'estate', 'price', 'rs', 'information', 'visit', 'need', 'offer',
    'offering', 'come', 'enjoy','apartment','islamabad','details','property','flat','square','feet','bedroom',
    'bed','sq','ft','building','house'
}
custom_stopwords.update(real_estate_noise)

# 3. Configure the Word Cloud
wordcloud = WordCloud(
    width=800, 
    height=400, 
    background_color='white',  # 'black' also looks sharp depending on preference
    stopwords=custom_stopwords,
    min_font_size=10,
    colormap='viridis'         # Try 'plasma', 'inferno', or 'coolwarm' for different vibes
).generate(text)

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
plt.tight_layout(pad=0)

# Display the figure seamlessly on the webpage
st.pyplot(fig)




# st.subheader("Word Cloud")

tab1, tab2, tab3 = st.tabs(["Price ", "Area", "No of Bedrooms"])

with tab1:
    st.subheader("Top 10 Most Expensive Locations")
    top_locations = (
        data.groupby('Main Location')["Price(Cr)"]
        .mean()
        .sort_values(ascending=False) # Optional: Makes the bar plot look cleaner (highest to lowest)
        .head(10)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=top_locations, 
        x='Main Location', 
        y="Price(Cr)", 
        ax=ax, 
        palette='viridis'  # Clean color theme matching your word cloud!
    )

    # Rotate labels slightly so location names (like 'Gulberg Green', 'DHA Phase') don't overlap
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Main Location")
    plt.ylabel("Average Price (Crore)")
    plt.tight_layout() # Prevents cut-off labels

    # --- 4. Render the plot in your Streamlit app ---
    st.pyplot(fig)
    

with tab2:
    st.subheader("📍 Top 10 Locations by Average Flat Size")

    # st.write("Comparing the average area size (in Marlas) for data across different main locations.")

    top_locations = (
        data.groupby('Main Location')["Area(Marla)"]
        .mean()
        .sort_values(ascending=False) # Optional: Makes the bar plot look cleaner (highest to lowest)
        .head(10)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=top_locations, 
        x='Main Location', 
        y="Area(Marla)", 
        ax=ax, 
        palette='viridis'  # Clean color theme matching your word cloud!
    )

    # Rotate labels slightly so location names (like 'Gulberg Green', 'DHA Phase') don't overlap
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Main Location")
    plt.ylabel("Average Size (Marla)")
    plt.tight_layout() # Prevents cut-off labels

    # --- 4. Render the plot in your Streamlit app ---
    st.pyplot(fig)
    

with tab3:
    # st.subheader("Top 10 Locations with most Bedrooms")
    st.subheader("📍 Top 10 Locations by Average Bedroom no.")

    # st.write("Comparing the average area size (in Marlas) for data across different main locations.")

    top_locations = (
        data.groupby('Main Location')["Bedroom(s)"]
        .mean()
        .sort_values(ascending=False) # Optional: Makes the bar plot look cleaner (highest to lowest)
        .head(10)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=top_locations, 
        x='Main Location', 
        y="Bedroom(s)", 
        ax=ax, 
        palette='viridis'  # Clean color theme matching your word cloud!
    )

    # Rotate labels slightly so location names (like 'Gulberg Green', 'DHA Phase') don't overlap
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Main Location")
    plt.ylabel("Average No of Bedrooms")
    plt.tight_layout() # Prevents cut-off labels

    # --- 4. Render the plot in your Streamlit app ---
    st.pyplot(fig)




st.subheader("Property Era with respect to Area and Price")
st.write("---")
st.write("**🏛️ Vintage:** Built in 1999 or earlier.")
st.write("**🏡 Established:** Built between 2000 and 2009.")
st.write("**🌆 Modern Era:** Built between 2010 and 2019.")
st.write("**✨ Recently Built:** Built between 2020 and 2024.")
st.write("**🆕 Brand New (2025):** Built in 2025.")
st.write("**🚀 Future:** Built in 2026 and beyond.")
# 1. Let the user choose between Price or Area
choice = st.selectbox("Select Metric:", ["Price(Cr)", "Area(Marla)"])

# 2. Get the average grouped by 'Property era' based on user choice
era_data = data.groupby('Property era')[choice].mean().reset_index()

# 3. Create and show the simple plot
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=era_data, x='Property era', y=choice, ax=ax)

# 4. Display on your Streamlit website
st.pyplot(fig)