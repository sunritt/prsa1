import streamlit as st
import os
from PIL import Image
from llm_pipeline import get_reviews_from_db, analyze_reviews

# Page config
st.set_page_config(page_title="Dynamic Analyzer", page_icon="✨", layout="wide")

# Custom CSS for modern design
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0b0f19;
        color: #e0e0e0;
    }
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    .stSelectbox label {
        color: #a0aabf;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease 0s;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        color: white;
        border: none;
    }
    
    /* Analysis container */
    .analysis-container {
        background-color: #151b2b;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #2d3748;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ Dynamic Product Analyzer")
st.markdown("Select a product to fetch its latest reviews from MongoDB and generate fresh AI insights.")

# Get product list from images directory
def get_product_list():
    if os.path.exists('images'):
        files = os.listdir('images')
        products = [os.path.splitext(f)[0] for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
        products = [p for p in products if p != 'logo']
        return sorted(products)
    return ["10Wbatten", "12WB22", "1WDL20"]

products = get_product_list()

col1, col2 = st.columns([1, 2])

with col1:
    selected_product = st.selectbox("Select a Product", products)
    
    # Display product image
    img_path_jpg = f"images/{selected_product}.jpg"
    img_path_png = f"images/{selected_product}.png"
    if os.path.exists(img_path_jpg):
        image = Image.open(img_path_jpg)
        st.image(image, use_container_width=True)
    elif os.path.exists(img_path_png):
        image = Image.open(img_path_png)
        st.image(image, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("Generate Fresh Analysis")

with col2:
    if generate_btn:
        with st.spinner(f"🔍 Querying MongoDB for {selected_product} reviews..."):
            reviews = get_reviews_from_db(selected_product, limit=200)
            
        if not reviews:
            st.error("No reviews found for this product in the database.")
        else:
            st.success(f"✅ Retrieved {len(reviews)} reviews. Analyzing with Gemini...")
            with st.spinner("🧠 Synthesizing insights..."):
                analysis = analyze_reviews(selected_product, reviews)
                
                # Display analysis in a formatted container
                st.markdown("### Analysis Results")
                st.markdown(analysis)
    else:
        st.info("👈 Select a product and click 'Generate Fresh Analysis' to begin.")
