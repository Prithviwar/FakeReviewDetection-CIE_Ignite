import streamlit as st
from utils import score_review

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fake Review Detection System",
    layout="centered"
)

# -----------------------------
# Title & Description
# -----------------------------
st.title("🕵️ Fake Review Detection System")
st.markdown(
    """
    This system analyzes online product reviews using **machine learning**
    and **rule-based scoring** to detect potentially fake or suspicious reviews.
    """
)

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
review_text = st.text_area(
    "Enter the review text:",
    height=150,
    placeholder="Paste the review here..."
)

rating = st.slider(
    "Select the product rating:",
    min_value=1,
    max_value=5,
    value=5
)

analyze_btn = st.button("🔍 Analyze Review")

# -----------------------------
# Analysis Output
# -----------------------------
if analyze_btn:
    if review_text.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        result = score_review(review_text, rating)

        score = result["score"]
        flag = result["flag"]
        probability = result["ml_fake_probability"]
        reasons = result["reasons"]

        st.divider()

        # -----------------------------
        # Flag Display
        # -----------------------------
        if flag == "GREEN":
            st.success("🟢 **Status:** Review appears genuine")
        elif flag == "YELLOW":
            st.warning("🟡 **Status:** Review is suspicious")
        else:
            st.error("🔴 **Status:** Review is likely fake")

        # -----------------------------
        # Metrics & Visualizations
        # -----------------------------
        st.write("### AI Analysis Breakdown")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Heuristic Score", f"{score}/100")
            st.progress(max(0, min(100, score)) / 100) # Clamp between 0-1.0
            
        with col2:
            prob_pct = int(probability * 100)
            st.metric("ML Fake Likelihood", f"{prob_pct}%")
            st.progress(probability)

        st.divider()
        
        # -----------------------------
        # Explanation
        # -----------------------------
        st.write("#### 🧠 Decision Reasoning")
        if reasons:
            for reason in reasons:
                st.info(f"🚩 {reason}")
        else:
            st.success("✅ No suspicious linguistic or structural patterns detected.")
