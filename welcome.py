import streamlit as st

# 1. Set page config (added layout="wide" to make it feel like a real dashboard)
st.set_page_config(
    page_title="Real Estate Hub",
    page_icon="🏠",
    layout="wide" 
)

# 2. Hero Section
st.title("Welcome to the 🏡 Real Estate Hub")
st.markdown("#### *Your AI-powered assistant for property valuation, market trends, and smart recommendations.*")
st.write("👈 **Open the sidebar** to select a tool and start exploring.")

st.divider()

# 3. Feature "Cards" using columns and colored boxes
st.subheader("🧭 What you can do here:")
st.write("") # Adds a tiny bit of breathing room

col1, col2 = st.columns(2)

with col1:
    st.info("""
    ### 🔮 Price Prediction
    Leverage our machine learning models to get accurate price estimates for both flats and houses based on current market data.
    """)
    
    st.success("""
    ### 📊 Market Analysis
    Dive deep into geographic price distributions, location-based trends, and property sizes with interactive maps and charts.
    """)

with col2:
    st.warning("""
    ### 🤖 Smart Recommendations
    Looking for the perfect spot? Let our AI match you with the best locations and buildings based on your budget and preferred amenities.
    """)

    
st.divider()

# Optional Footer
st.markdown("<p style='text-align: center; color: gray;'>Built with Streamlit • Real Estate Market Data 2025-2026</p>", unsafe_allow_html=True)