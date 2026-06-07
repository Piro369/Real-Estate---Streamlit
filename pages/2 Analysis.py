import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ==========================================
# CACHING HEAVY OPERATIONS (SPEED UPGRADES)
# ==========================================

@st.cache_data
def load_data():
    """Loads CSVs once and keeps them in memory."""
    flats = pd.read_csv('flats.csv', index_col=0)
    houses = pd.read_csv('houses.csv', index_col=0)
    flats_coord = pd.read_csv('flats_lat_long.csv', index_col=0)
    houses_coord = pd.read_csv('houses_lat_long.csv', index_col=0)

    # Pre-filter outliers so it doesn't happen on every click
    houses = houses[houses['Price(Cr)'] < 150]
    houses_coord = houses_coord[houses_coord['Price(Cr)'] < 150]
    
    return flats, houses, flats_coord, houses_coord

@st.cache_data
def get_wordcloud_text(descriptions):
    """Joins text descriptions once per property type."""
    return " ".join(desc for desc in descriptions.dropna())


# Load data using the cached function
flats, houses, flats_coord, houses_coord = load_data()


# ==========================================
# MAIN UI
# ==========================================

st.title("🏡 Real Estate Market Analysis")
property_type = st.radio("Select Property Type", ['House', 'Flat'])

# Dynamic naming suffix (e.g., "for Houses" or "for Flats")
prop_suffix = f"for {property_type}s"

if property_type == 'Flat':
    data = flats
    data_coord = flats_coord
else:
    # Note: Since we already filtered to < 150 above, we don't need to filter again here.
    data = houses
    data_coord = houses_coord

# --- SECTION 1: MAP ---
st.subheader(f"📍 Geographic Price Distribution {prop_suffix}")

data_coord_clean = data_coord.dropna(subset=['Latitude', 'Longitude', 'Price(Cr)', 'Area(Marla)']) 
grouped_df = data_coord_clean.groupby(['Main Location', 'Latitude', 'Longitude']).agg(
    Average_Price=('Price(Cr)', 'mean'),   
    Average_Area=('Area(Marla)', 'mean'),
    Total_Listings=('Price(Cr)', 'count')
).reset_index()

grouped_df['Average_Price'] = grouped_df['Average_Price'].round(2)
grouped_df['Average_Area'] = grouped_df['Average_Area'].round(1)

# Build the Density Map
fig = px.scatter_mapbox(
    grouped_df,
    lat='Latitude',
    lon='Longitude',
    color='Average_Price',           
    size='Total_Listings',            
    color_continuous_scale="Plasma",
    size_max=30,
    center=dict(lat=33.6844, lon=73.0479),
    zoom=10.5,
    mapbox_style="carto-darkmatter",
    hover_name="Main Location",
    hover_data={
        'Latitude': False, 
        'Longitude': False, 
        'Average_Price': True, 
        'Average_Area': True,
        'Total_Listings': False
    },
    labels={
        'Average_Price': 'Avg Price (Cr)',
        'Average_Area': 'Avg Area (Marla)',
    },
    title="Average Property Price per Location in Crores"
)

fig.update_layout(margin={"r":0, "t":40, "l":0, "b":0})
st.plotly_chart(fig, use_container_width=True)

st.write("---")

# --- SECTION 2: WORD CLOUD ---
st.subheader(f"🔤 Most Frequent Keywords in {property_type} Descriptions")

# Uses the cached text processing function!
text = get_wordcloud_text(data['Description'])

custom_stopwords = set(STOPWORDS)
real_estate_noise = {
    'sale', 'available', 'detail', 'project', 'option', 'best', 'prime', 
    'location', 'ideal', 'contact', 'us', 'please', 'today', 'make', 'one', 
    'view', 'brand', 'new', 'looking', 'find', 'feel', 'free', 'investment', 
    'investor', 'opportunity', 'ready', 'value', 'quality', 'high', 'type',
    'real', 'estate', 'price', 'rs', 'information', 'visit', 'need', 'offer',
    'offering', 'come', 'enjoy','apartment','islamabad','details','property','flat','square','feet','bedroom',
    'bed','sq','ft','building','house','marla'
}
custom_stopwords.update(real_estate_noise)

wordcloud = WordCloud(
    width=800, 
    height=400, 
    background_color='white', 
    stopwords=custom_stopwords,
    min_font_size=10,
    colormap='viridis'        
).generate(text)

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
plt.tight_layout(pad=0)

st.pyplot(fig)

st.write("---")

# --- SECTION 3: TABS ---
st.subheader(f"📊 Top 10 Location Breakdowns {prop_suffix}")

tab1, tab2, tab3 = st.tabs(["💰 Price Breakdown", "📐 Area Breakdown", "🛏️ Bedroom Count"])

with tab1:
    st.markdown(f"### Top 10 Most Expensive Locations {prop_suffix}")
    top_locations = (
        data.groupby('Main Location')["Price(Cr)"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_locations, x='Main Location', y="Price(Cr)", ax=ax, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Main Location")
    plt.ylabel("Average Price (Crore)")
    plt.tight_layout() 
    st.pyplot(fig)
    

with tab2:
    st.markdown(f"### Top 10 Locations by Average {property_type} Size")
    top_locations = (
        data.groupby('Main Location')["Area(Marla)"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_locations, x='Main Location', y="Area(Marla)", ax=ax, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Main Location")
    plt.ylabel("Average Size (Marla)")
    plt.tight_layout()
    st.pyplot(fig)
    

with tab3:
    st.markdown(f"### Top 10 Locations by Average Room Count")
    top_locations = (
        data.groupby('Main Location')["Bedroom(s)"]
        .mean()
        .sort_values(ascending=False) 
        .head(10)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_locations, x='Main Location', y="Bedroom(s)", ax=ax, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Main Location")
    plt.ylabel("Average No of Bedrooms")
    plt.tight_layout()
    st.pyplot(fig)

st.write("---")

# --- SECTION 4: ERA ANALYSIS ---
st.subheader(f"⏳ {property_type} Valuation & Size by Construction Era")
st.write("**🏛️ Vintage:** Built in 1999 or earlier.")
st.write("**🏡 Established:** Built between 2000 and 2009.")
st.write("**🌆 Modern Era:** Built between 2010 and 2019.")
st.write("**✨ Recently Built:** Built between 2020 and 2024.")
st.write("**🆕 Brand New (2025):** Built in 2025.")
st.write("**🚀 Future:** Built in 2026 and beyond.")

choice = st.selectbox("Select Metric:", ["Price(Cr)", "Area(Marla)"])
era_data = data.groupby('Property era')[choice].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=era_data, x='Property era', y=choice, ax=ax, palette='viridis')
plt.ylabel(f"Average {choice}")
plt.tight_layout()

st.pyplot(fig)