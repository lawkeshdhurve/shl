import os
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time  # Optional

@st.cache_data(ttl=3600)
def load_assessments():
    try:
        json_path = os.path.join(os.path.dirname(__file__), 'assessments.json')
        data = pd.read_json(json_path)
        return data
    except Exception as e:
        st.error(f"Failed to load assessments from JSON: {e}")
        return pd.DataFrame(columns=['title', 'description'])

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def recommend_assessments(job_text, df, model):
    # Add a check here *just in case* an invalid DataFrame slips through
    if 'description' not in df.columns or df.empty:
        st.error("Cannot generate recommendations: Assessment data is invalid or empty.")
        return pd.DataFrame()  # Return empty DataFrame

    job_embedding = model.encode([job_text])
    # Ensure descriptions are strings
    descriptions = df['description'].astype(str).tolist()
    assessment_embeddings = model.encode(descriptions)

    similarities = cosine_similarity(job_embedding, assessment_embeddings)[0]

    # Use .copy() when adding the new column to avoid SettingWithCopyWarning
    df_results = df.copy()
    df_results['similarity'] = similarities

    return df_results.sort_values(by='similarity', ascending=False).head(5)

# --- Streamlit UI ---
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🔍 SHL AI Assessment Recommender")
st.markdown("Paste a job description or natural language query, and get matched SHL assessments.")

job_input = st.text_area(
    "📝 Job Description or Query",
    height=150,
    placeholder="e.g., 'Software Engineer with Python and cloud experience', or paste a full job description..."
)

if st.button("🔎 Recommend Assessments"):
    if not job_input.strip():
        st.warning("Please enter a job description or query.")
    else:
        with st.spinner("Loading assessments and analyzing query..."):
            # Load data first
            df_assessments = load_assessments()

            # Check if loading was successful before proceeding
            if df_assessments.empty or 'description' not in df_assessments.columns:
                st.error("Failed to load assessment data. Cannot provide recommendations.")
            else:
                try:
                    # Load model only if data is good
                    model = load_model()
                    # Get recommendations
                    results = recommend_assessments(job_input, df_assessments, model)

                    if not results.empty:
                        st.subheader("✅ Top Matching SHL Assessments")
                        for idx, row in results.iterrows():
                            st.markdown(f"**{row['title']}**")
                            # Only show description if it's not the placeholder
                            if row['description'] != "No Description Found":
                                st.markdown(f"🧠 _{row['description']}_")
                            else:
                                st.markdown("_No description available._")
                            st.markdown(f"📊 Similarity Score: `{row['similarity']:.3f}`")  # More precision
                            st.markdown("---")
                    else:
                        st.info("No matching assessments found based on the similarity calculation, or an error occurred during recommendation.")

                except Exception as e:
                    st.error(f"An error occurred during model loading or recommendation: {e}")
                    import traceback
                    st.error("Traceback:")
                    st.code(traceback.format_exc())

# Add a footer or disclaimer
st.markdown("---")
st.caption("Disclaimer: Assessment data is scraped from the SHL website and may change. Recommendations are based on semantic similarity.")              