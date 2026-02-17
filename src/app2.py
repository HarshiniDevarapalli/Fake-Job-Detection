# ==========================================
# 📘 Job Scam Detection Streamlit App
# ==========================================
import streamlit as st
import pandas as pd
import joblib
import requests
import tldextract

# ----------------------------
# Load your trained model
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("fake_job_model_v3.pkl")
    return model

model = load_model()

# ----------------------------
# Helper functions
# ----------------------------
def check_website_status(url):
    """Check if a website is reachable."""
    try:
        if not isinstance(url, str) or not url.strip():
            return 0
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        r = requests.head(url, timeout=5, allow_redirects=True)
        return int(r.status_code == 200)
    except:
        return 0

def extract_domain_features(row):
    """Extract domain, email, and company match features."""
    website = str(row.get('company_profile', '')).lower()
    email = str(row.get('company_email', '')).lower()
    company = str(row.get('company', '')).lower().replace(" ", "")

    extracted = tldextract.extract(website)
    domain = extracted.domain
    suffix = extracted.suffix

    free_domains = ['gmail', 'yahoo', 'hotmail', 'outlook', 'aol', 'protonmail']
    is_free_email = any(f"@{d}." in email for d in free_domains)
    has_website = bool(website.strip()) and ("http" in website or "." in website)
    domain_match = company in domain if company else False
    website_active = check_website_status(website) if has_website else 0

    return pd.Series({
        'has_website': int(has_website),
        'free_email': int(is_free_email),
        'domain_match': int(domain_match),
        'website_active': int(website_active),
        'suffix': suffix
    })

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Fake Job Detector", page_icon="🔍", layout="centered")

st.title("🔍 Fake Job Posting Detector")
st.markdown("### Enter job posting details below to check if it's **real or fake**.")
st.write("The model uses NLP + company website/email analysis to detect scams.")

with st.form("job_input"):
    title = st.text_input("Job Title")
    description = st.text_area("Job Description", height=200)
    company = st.text_input("Company Name")
    website = st.text_input("Company Website / Profile URL")
    email = st.text_input("Company Email Address")
    submitted = st.form_submit_button("🔎 Check Job Posting")

# ----------------------------
# Prediction Logic
# ----------------------------
if submitted:
    if not title or not description:
        st.warning("⚠️ Please fill in at least the job title and description.")
    else:
        st.info("⏳ Analyzing posting... Please wait.")

        # Prepare input data
        df = pd.DataFrame([{
            "title": title,
            "description": description,
            "company": company,
            "company_profile": website,
            "company_email": email
        }])

        # Combine text fields
        df["text"] = (df["title"].fillna('') + " " + df["description"].fillna('')).str.strip()

        # Extract domain and email-based features
        domain_features = df.apply(extract_domain_features, axis=1)
        df = pd.concat([df, domain_features], axis=1)

        # Extract individual values for rules
        has_website = bool(df.loc[0, 'has_website'])
        free_email = bool(df.loc[0, 'free_email'])
        domain_match = bool(df.loc[0, 'domain_match'])
        website_active = bool(df.loc[0, 'website_active'])

        # --- Run ML model ---
        base_prob = model.predict_proba(df)[0][1]  # fake probability

        # --- Risk logic ---
        risk_score = 0
        reasons = []

        if not has_website:
            risk_score += 2
            reasons.append("No company website provided")
        if free_email:
            risk_score += 3
            reasons.append("Free email domain used")
        if not domain_match:
            risk_score += 2
            reasons.append("Domain doesn’t match company name")
        if not website_active:
            risk_score += 3
            reasons.append("Website is inactive or unreachable")

        # --- Adjust fake probability ---
        adjusted_fake_prob = min(base_prob + (risk_score * 0.1), 1.0)

        # --- Override logic ---
        if risk_score >= 4:
            final_pred = 1  # Force fake
        elif adjusted_fake_prob > 0.5:
            final_pred = 1
        else:
            final_pred = 0

        # ----------------------------
        # Output section
        # ----------------------------
        st.subheader("🧠 Model Prediction:")
        if final_pred == 1:
            st.error(f"🚨 This job posting appears **FAKE / SCAM** "
                     f"(Adjusted Fake Probability: {adjusted_fake_prob:.2f})")
            if reasons:
                st.markdown("**Reasons:** " + ", ".join(reasons))
        elif adjusted_fake_prob > 0.35:
            st.warning(f"⚠️ This job may be **SUSPICIOUS** "
                       f"(Fake Probability: {adjusted_fake_prob:.2f})")
            if reasons:
                st.markdown("**Caution:** " + ", ".join(reasons))
        else:
            st.success(f"✅ This job posting appears **LEGITIMATE** "
                       f"(Fake Probability: {adjusted_fake_prob:.2f})")

        st.markdown("---")
        st.caption("Model: RandomForest + Risk Overrides | Features: TF-IDF + Domain + Email + Website Checks")
