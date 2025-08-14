
import streamlit as st

# Title
st.title("Fake News Detector")

# Input text
user_input = st.text_area("Enter the news text here:")

# Dummy prediction logic for now
if st.button("Check"):
    if "fake" in user_input.lower():
        st.error("🚨 This news might be FAKE!")
    elif user_input.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        st.success("✅ This news looks REAL!")
