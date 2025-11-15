import streamlit as st
import joblib
import time

# --- Configuration ---
MODEL_PATH = 'Model/best_tfidf_sentiment_model.joblib'
st.set_page_config(
    page_title="TF-IDF Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Model Loading with Caching ---

@st.cache_resource
def load_sentiment_model():
    """
    Loads the trained model pipeline using joblib.
    Uses st.cache_resource to load the model only once, speeding up the app.
    """
    with st.spinner(f"Loading model: {MODEL_PATH}..."):
        time.sleep(0.5)
        try:
            model = joblib.load(MODEL_PATH)
            st.success("Model loaded successfully from disk.")

        
            # --------------------------------------------------------------------------

        except FileNotFoundError:
            st.error(f"Error: Model file '{MODEL_PATH}' not found.")
            st.warning("Using a mock model. Please ensure your saved model is in the root directory.")
        except Exception as e:
            st.error(f"An unexpected error occurred during model loading: {e}")
            

    return model

# --- Main Streamlit Application ---
def main():
    st.title("Sentiment Analysis Deployment 🚀")
    st.markdown("Enter a piece of text (e.g., a review, comment, or statement) below to classify its sentiment as Positive or Negative.")
    st.markdown("The model uses TF-IDF so the accuracy is not too good as NN models ,it is only 85% using LogisticRegression on CV. ")

    # Load the model (cached)
    model = load_sentiment_model()

    st.markdown("---")

    # Text Input Area
    user_input = st.text_area(
        "Enter Text Here:",
        "This project is running much faster now that I optimized the pipeline!",
        height=150
    )

    # Prediction Button
    if st.button("Analyze Sentiment", type="primary"):
        if not user_input or user_input.strip() == "":
            st.error("Please enter some text to analyze.")
            return

        with st.spinner("Classifying sentiment..."):
            time.sleep(0.5) # Simulate quick prediction time

            # Prediction
            # The pipeline handles both TF-IDF transformation and classification
            prediction_array = model.predict([user_input])
            prediction = prediction_array[0]

            # Display Result
            if prediction == 1:
                st.balloons()
                st.markdown(
                    "<div style='background-color:#047857; padding:20px; border-radius:10px; text-align:center;'>"\
                    "<h2 style='color:white; margin:0;'>Sentiment: POSITIVE 😊</h2>"\
                    "</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='background-color:#b91c1c; padding:20px; border-radius:10px; text-align:center;'>"\
                    "<h2 style='color:white; margin:0;'>Sentiment: NEGATIVE 😔</h2>"\
                    "</div>",
                    unsafe_allow_html=True
                )

    st.markdown("---")
    st.info("Model Info: This app is powered by the highly optimized TF-IDF + Logistic Regression/LinearSVC Pipeline Abhishek/I developed.")

if __name__ == "__main__":
    main()