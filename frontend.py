
import streamlit as st
import requests
from PIL import Image

st.set_page_config(
    page_title="EcoVision-AI",
    page_icon="♻️",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/predict"

st.title("♻️ EcoVision-AI")
st.subheader("AI-Powered Waste Classification")

st.write(
    "Upload a waste image and let our AI model "
    "identify its category."
)

uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Waste Image",
        use_container_width=True
    )

    if st.button("Identify Waste", type="primary"):

        with st.spinner("AI is analyzing your image..."):

            try:
                uploaded_file.seek(0)

                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file,
                            uploaded_file.type
                        )
                    },
                    timeout=120
                )

                response.raise_for_status()
                result = response.json()

                if "error" in result:
                    st.error(result["error"])

                else:
                    category = result["category"]
                    confidence = result["confidence"]

                    st.success("Waste identification completed!")

                    st.subheader("Prediction Result")

                    st.write("Waste Category:", category.title())

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence}%"
                    )

                    st.progress(
                        max(0.0, min(1.0, confidence / 100))
                    )

                    recommendations = {
                        "plastic": "Send clean plastic to an appropriate plastic recycling facility.",
                        "paper": "Keep paper dry and send it for paper recycling.",
                        "glass": "Separate glass and take it to a suitable glass recycling point.",
                        "metal": "Separate metal waste for metal recycling."
                    }

                    advice = recommendations.get(
                        category.lower(),
                        "Check local waste disposal guidelines."
                    )

                    st.subheader("♻️ Recycling Recommendation")
                    st.info(advice)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to backend. "
                    "Make sure FastAPI is running."
                )

            except Exception as e:
                st.error(f"Something went wrong: {e}")

st.divider()

st.caption("EcoVision-AI | Smart Waste Management Project")

from auth import register_user, login_user

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("♻️ EcoVision-AI")
    st.subheader("User Account")

    mode = st.radio(
        "Choose an option",
        ["Login", "Register"]
    )

    username = st.text_input("Username")
    password = st.text_input(
        "Password",
        type="password"
    )

    if mode == "Register":

        if st.button("Create Account"):
            success, message = register_user(
                username, password
            )

            if success:
                st.success(message)
            else:
                st.error(message)

    else:

        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password.")

    st.stop()

st.sidebar.success(
    f"Welcome, {st.session_state.username}!"
)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()
