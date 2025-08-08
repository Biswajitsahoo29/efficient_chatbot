import os
import streamlit as st
import torch
import speech_recognition as sr
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import tempfile

# ✅ Load AI Models
sentiment_analyzer = pipeline("sentiment-analysis")
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ✅ Load DialoGPT for Chatbot
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# ✅ Predefined Mental Health Responses
mental_health_responses = {
    "anxious": "Take deep breaths. I'm here for you. Try grounding techniques.",
    "stressed": "Step away for a bit, try meditation or deep breathing.",
    "depressed": "You're not alone. Seeking help from a therapist could be helpful.",
    "lonely": "Connect with someone you trust or engage in an enjoyable activity.",
    "sad": "It's okay to feel sad. Express your emotions freely.",
    "tired": "Ensure you're well-rested and hydrated. Take small breaks.",
    "worried": "Try focusing on what you can control. Writing it down may help."
}

# ✅ Sentiment-Based Response Mapping
sentiment_responses = {
    "POSITIVE": "That's great to hear! Keep up the positive vibes.",
    "NEGATIVE": "I'm here for you. It's okay to feel this way. Do you want to talk about it?",
    "NEUTRAL": "Thanks for sharing. Would you like to talk more about how you're feeling?"
}

def detect_mental_health_issues(user_input):
    user_input = user_input.lower()
    for keyword in mental_health_responses:
        if keyword in user_input:
            return mental_health_responses[keyword]
    return None

def analyze_sentiment(user_input):
    result = sentiment_analyzer(user_input)[0]
    return result['label']

def chatbot_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Error: Could not understand audio."
    except sr.RequestError:
        return "Error: Could not request results."

def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

def smart_response(user_input):
    corpus = list(mental_health_responses.keys())
    embeddings = embedder.encode(corpus + [user_input])
    similarities = cosine_similarity([embeddings[-1]], embeddings[:-1])
    best_match_idx = np.argmax(similarities)
    if similarities[0][best_match_idx] > 0.5:
        return mental_health_responses[corpus[best_match_idx]]
    return None

def main():
    st.title("🧠 AI Mental Health Chatbot")
    st.write("Chat with an AI-powered assistant that understands your emotional state.")

    # ✅ Session state to store full conversation
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:", "")

    if user_input:
        # Try keyword-based or similarity-based responses
        response = detect_mental_health_issues(user_input) or smart_response(user_input)

        # If no predefined response found, use DialoGPT
        if not response:
            response = chatbot_response(user_input)

        # Sentiment fallback only if nothing meaningful generated
        if not response or response.strip() == "":
            sentiment = analyze_sentiment(user_input)
            response = sentiment_responses.get(sentiment, sentiment_responses["NEUTRAL"])

        # Save to history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    # ✅ Display full conversation history
    st.subheader("🧾 Chat History")
    for sender, msg in st.session_state.chat_history:
        st.markdown(f"**{sender}:** {msg}")

    # 🔊 Voice Input
    st.subheader("🔊 Voice Input")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if uploaded_file:
        text_output = speech_to_text(uploaded_file)
        st.write(f"**Converted Text:** {text_output}")

    # 🗣️ Text-to-Speech
    st.subheader("🗣️ Text-to-Speech")
    if st.button("Speak Last Bot Response"):
        if st.session_state.chat_history:
            last_bot_responses = [msg for sender, msg in st.session_state.chat_history if sender == "Bot"]
            if last_bot_responses:
                last_response = last_bot_responses[-1]
                speech_file = text_to_speech(last_response)
                st.audio(speech_file, format="audio/mp3")

if __name__ == "__main__":
    main()
