import streamlit as st
import joblib
import plotly.graph_objects as go
import base64


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SMS Fraud Detection",
    page_icon="🔐",
    layout="centered"
)


# =========================================================
# BACKGROUND IMAGE
# =========================================================

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


background_image = get_base64_image("assets/download.png")


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================

vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("spam_model.pkl")


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    f"""
    <style>

    /* =====================================================
       BACKGROUND
       ===================================================== */

    .stApp {{
        background-image:
            linear-gradient(
                rgba(20, 45, 85, 0.48),
                rgba(20, 45, 85, 0.55)
            ),
            url("data:image/png;base64,{background_image}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}


    /* =====================================================
       PAGE WIDTH
       ===================================================== */

    .block-container {{
        max-width: 850px;
        padding-top: 2.5rem;
        padding-bottom: 5rem;
    }}


    /* =====================================================
       MAIN TITLE
       ===================================================== */

    h1 {{
        color: #111827 !important;

        font-family:
            Arial,
            Helvetica,
            sans-serif !important;

        font-size: 42px !important;

        font-weight: 800 !important;

        text-align: center;

        letter-spacing: -1px;

        margin-bottom: 8px !important;
    }}


    /* =====================================================
       SUBTITLE
       ===================================================== */

    .subtitle {{
        color: #183b63;

        font-family:
            Arial,
            Helvetica,
            sans-serif;

        font-size: 19px;

        font-weight: 600;

        text-align: center;

        margin-bottom: 30px;
    }}


    /* =====================================================
       CHAT AREA
       ===================================================== */

    [data-testid="stChatMessage"] {{
        border-radius: 16px;
        margin-bottom: 12px;
    }}


    /* =====================================================
       CHAT INPUT
       ===================================================== */

    [data-testid="stChatInput"] {{
        border-radius: 16px;
    }}


    [data-testid="stChatInput"] textarea {{
        font-size: 16px !important;
    }}


    /* =====================================================
       WELCOME TEXT
       ===================================================== */

    .welcome-title {{
        color: #0f172a;

        font-size: 24px;

        font-weight: 700;

        margin-bottom: 8px;
    }}


    .welcome-text {{
        color: #1e3a5f;

        font-size: 19px;

        font-weight: 500;

        line-height: 1.7;
    }}


    /* =====================================================
       RESULT BOX
       ===================================================== */

    .result-box {{
        background: rgba(255, 255, 255, 0.95);

        color: #111827;

        padding: 17px 20px;

        border-radius: 14px;

        font-size: 19px;

        font-weight: 700;

        margin-top: 8px;

        margin-bottom: 18px;

        box-shadow:
            0 5px 18px rgba(0, 0, 0, 0.18);
    }}


    /* =====================================================
       MODEL ANALYSIS
       ===================================================== */

    .analysis-title {{
        color: #111827;

        font-size: 22px;

        font-weight: 800;

        margin-top: 15px;

        margin-bottom: 10px;
    }}


    /* =====================================================
       SAFETY HEADING
       ===================================================== */

    .safety-title {{
        color: #9a5b00;

        font-size: 23px;

        font-weight: 800;

        margin-top: 22px;

        margin-bottom: 12px;
    }}


    /* =====================================================
       SAFETY TEXT
       ===================================================== */

    .safety-box {{
        background: rgba(255, 255, 255, 0.95);

        color: #111827;

        font-size: 18px;

        font-weight: 500;

        line-height: 1.8;

        padding: 20px 24px;

        border-radius: 15px;

        border-left: 6px solid #f59e0b;

        box-shadow:
            0 5px 18px rgba(0, 0, 0, 0.18);
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.title("SMS Fraud Detection")

st.markdown(
    """
    <div class="subtitle">
        Check whether your SMS is likely to be spam or legitimate.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# INITIAL WELCOME MESSAGE
# =========================================================

if len(st.session_state.messages) == 0:

    with st.chat_message("assistant"):

        st.markdown(
            """
            <div class="welcome-title">
                Hello! 👋
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="welcome-text">
                Send me an SMS message and I'll analyze it to
                determine whether it is likely to be
                <b>SPAM</b> or <b>LEGITIMATE</b>.
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# DISPLAY PREVIOUS CHAT
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        # -------------------------------------------------
        # USER MESSAGE
        # -------------------------------------------------

        if message["type"] == "user":

            st.write(message["content"])


        # -------------------------------------------------
        # BOT RESULT
        # -------------------------------------------------

        elif message["type"] == "result":

            if message["prediction"] == 1:

                st.markdown(
                    """
                    <div class="result-box">
                        🚨 This message is likely to be SPAM.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="result-box">
                        ✅ This message appears to be LEGITIMATE.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ---------------------------------------------
            # MODEL ANALYSIS
            # ---------------------------------------------

            st.markdown(
                """
                <div class="analysis-title">
                    📊 Model Analysis
                </div>
                """,
                unsafe_allow_html=True
            )


            # ---------------------------------------------
            # GRAPH
            # ---------------------------------------------

            fig = go.Figure()


            # SPAM BAR

            fig.add_trace(
                go.Bar(
                    y=["SPAM"],
                    x=[message["spam_probability"]],
                    orientation="h",

                    text=[
                        f'{message["spam_probability"]:.1f}%'
                    ],

                    textposition="auto",

                    marker_color="#7c3aed"
                )
            )


            # HAM BAR

            fig.add_trace(
                go.Bar(
                    y=["HAM"],
                    x=[message["ham_probability"]],
                    orientation="h",

                    text=[
                        f'{message["ham_probability"]:.1f}%'
                    ],

                    textposition="auto",

                    marker_color="#64748b"
                )
            )


            fig.update_layout(

                xaxis=dict(
                    range=[0, 100],
                    title="Model Probability (%)"
                ),

                yaxis=dict(
                    title=""
                ),

                barmode="group",

                height=240,

                margin=dict(
                    l=20,
                    r=20,
                    t=15,
                    b=35
                ),

                paper_bgcolor="#172554",

                plot_bgcolor="#172554",

                font=dict(
                    color="white"
                ),

                showlegend=False
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
                key=message["chart_key"]
            )


            # ---------------------------------------------
            # SAFETY ADVICE — SPAM ONLY
            # ---------------------------------------------

            if message["prediction"] == 1:

                st.markdown(
                    """
                    <div class="safety-title">
                        ⚠️ Don't Panic — Stay Safe
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Native Streamlit markdown.
                # No HTML here, so raw <div> / <br> code
                # cannot appear in the safety message.

                st.markdown(
                    """
                    <div class="safety-box">
                    """,
                    unsafe_allow_html=True
                )

                st.write(
                    "This message appears suspicious based on the "
                    "trained machine-learning model."
                )

                st.write(
                    "🔗 **Don't click any links** that may be included."
                )

                st.write(
                    "📞 **Don't call unknown numbers** mentioned in "
                    "the message."
                )

                st.write(
                    "🔐 **Never share OTPs, passwords, banking details, "
                    "or other personal information.**"
                )

                st.write(
                    "🗑️ If you don't recognize the sender, consider "
                    "ignoring, blocking, or deleting the message."
                )

                st.markdown(
                    """
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =========================================================
# CHAT INPUT
# =========================================================

user_message = st.chat_input(
    "Type or paste your SMS here..."
)


# =========================================================
# PROCESS NEW SMS
# =========================================================

if user_message:

    # -----------------------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "type": "user",
            "content": user_message
        }
    )


    # -----------------------------------------------------
    # TF-IDF
    # -----------------------------------------------------

    message_tfidf = vectorizer.transform(
        [user_message]
    )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(
        message_tfidf
    )[0]


    # -----------------------------------------------------
    # PROBABILITY
    # -----------------------------------------------------

    probabilities = model.predict_proba(
        message_tfidf
    )[0]


    ham_probability = probabilities[0] * 100

    spam_probability = probabilities[1] * 100


    # -----------------------------------------------------
    # SAVE RESULT
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",

            "type": "result",

            "prediction": int(prediction),

            "ham_probability":
                float(ham_probability),

            "spam_probability":
                float(spam_probability),

            "chart_key":
                f"chart_{len(st.session_state.messages)}"
        }
    )


    # -----------------------------------------------------
    # REFRESH
    # -----------------------------------------------------

    st.rerun()