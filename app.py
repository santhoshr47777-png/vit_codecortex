import streamlit as st
import pandas as pd

from src.threat_analyzer import analyze_email


st.set_page_config(
    page_title="ThreatLens X",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ ThreatLens X")
st.subheader("AI-Powered Cyber Threat Detection & Investigation")

st.markdown(
    """
    ThreatLens X combines machine learning, threat similarity,
    explainability, and anomaly detection to investigate suspicious emails.
    """
)

st.divider()

email_text = st.text_area(
    "📧 Enter an email for analysis",
    height=250,
    placeholder="Paste the email content here..."
)

analyze_button = st.button(
    "🔍 Analyze Threat",
    type="primary"
)

if analyze_button:

    if not email_text.strip():
        st.warning("Please enter an email first.")
        st.stop()

    with st.spinner("Analyzing threat..."):
        result = analyze_email(email_text)

    classification = result["classification"]
    dna = result["threat_dna"]
    similar = result["similar_threats"]
    novelty = result["novelty"]
    risk = result["risk"]

    st.divider()

    # -------------------------
    # Main result
    # -------------------------

    st.header("🎯 Detection Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Classification",
            classification["label"]
        )

    with col2:
        st.metric(
            "Model Confidence",
            f'{classification["confidence"]:.1f}%'
        )

    with col3:
        st.metric(
            "Risk Score",
            f'{risk["risk_score"]:.1f}/100'
        )

    st.info(
        f'Risk Severity: **{risk["severity"]}**'
    )

    # -------------------------
    # Threat DNA
    # -------------------------

    st.header("🧬 Threat DNA")

    if dna:
        dna_df = pd.DataFrame(dna)

        st.dataframe(
            dna_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("No significant threat features found.")

    # -------------------------
    # Similar threats
    # -------------------------

    st.header("🔎 Similar Historical Threats")

    if similar:

        similarity_data = []

        for item in similar:

            similarity_data.append({
                "Similarity": f'{item["similarity"]:.1f}%',
                "Historical Label": item["label"],
                "Email": item["text"]
            })

        similarity_df = pd.DataFrame(similarity_data)

        st.dataframe(
            similarity_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.write("No similar historical threats found.")

    # -------------------------
    # Novelty
    # -------------------------

    st.header("🕵️ Novelty Detection")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Novelty Score",
            f'{novelty["novelty_score"]:.1f}/100'
        )

    with col2:
        st.metric(
            "Pattern Status",
            novelty["status"]
        )

    st.caption(
        "Novelty score represents how unusual the email is relative "
        "to patterns learned from the reference dataset. It is not a probability."
    )

    # -------------------------
    # Risk breakdown
    # -------------------------

    st.header("⚠️ Risk Breakdown")

    risk_data = {
        "Component": [
            "Model Confidence",
            "Evidence",
            "Similarity",
            "Novelty"
        ],
        "Score": [
            risk["model_confidence"],
            risk["evidence_score"],
            risk["similarity_score"],
            risk["anomaly_score"]
        ]
    }

    risk_df = pd.DataFrame(risk_data)

    st.bar_chart(
        risk_df.set_index("Component")
    )

    # -------------------------
    # Analyst summary
    # -------------------------

    st.header("🧑‍💻 Analyst Summary")

    if classification["label"] == "SPAM":

        st.warning(
            "This email was classified as suspicious/spam. "
            "Review the Threat DNA, historical similarities, "
            "and novelty indicators before taking action."
        )

    else:

        st.success(
            "This email was classified as legitimate by the current model. "
            "The anomaly and similarity indicators should still be considered "
            "when investigating unusual messages."
        )

    st.divider()

    st.caption(
        "ThreatLens X — AI Cyber Threat Detection & Investigation Platform"
    )