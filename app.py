import os
import streamlit as st
import torch
import speech_recognition as sr
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import tempfile
import json
from datetime import datetime

# ✅ Load AI Models
@st.cache_resource
def load_models():
    # Explicitly specify the sentiment analysis model to avoid warnings
    sentiment_analyzer = pipeline("sentiment-analysis", 
                                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                                revision="af0f99b")
    
    # Load sentence transformer with error handling
    try:
        embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading sentence transformer: {e}")
        embedder = None
    
    # Load DialoGPT for Chatbot with proper tokenizer configuration
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    
    # Fix padding configuration for decoder-only model
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Set left padding for decoder-only models
    
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    
    return sentiment_analyzer, embedder, tokenizer, model

# Initialize models with error handling
try:
    sentiment_analyzer, embedder, tokenizer, model = load_models()
    st.success("✅ All models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# ✅ Enhanced Mental Health Training Data
mental_health_training_data = [
    {"input": "I feel anxious about everything", "response": "I understand anxiety can be overwhelming. Let's try some grounding techniques. Can you name 5 things you can see around you right now?"},
    {"input": "I'm having a panic attack", "response": "Focus on your breathing. Breathe in for 4 counts, hold for 4, breathe out for 4. You're safe. This feeling will pass."},
    {"input": "I feel so depressed", "response": "I hear you, and I'm sorry you're going through this. Depression is treatable. Have you considered speaking with a mental health professional?"},
    {"input": "I can't sleep at night", "response": "Sleep issues can really affect your wellbeing. Try establishing a bedtime routine, avoiding screens before bed, and consider relaxation techniques."},
    {"input": "I feel lonely and isolated", "response": "Loneliness is difficult. Even small connections can help - calling a friend, joining online communities, or volunteering. You're not alone."},
    {"input": "I'm stressed about work", "response": "Work stress is common. Try breaking tasks into smaller chunks, take regular breaks, and remember that your worth isn't defined by work productivity."},
    {"input": "I feel like giving up", "response": "These feelings are temporary, even though they feel overwhelming now. Please reach out to a crisis helpline or mental health professional immediately."},
    {"input": "I'm worried about my health", "response": "Health anxiety is understandable. While I can't provide medical advice, speaking with a healthcare provider about your concerns might help ease your worry."},
    {"input": "I feel angry all the time", "response": "Persistent anger can be exhausting. Identifying triggers, physical exercise, and talking to someone about underlying issues can help manage these feelings."},
    {"input": "I have no motivation", "response": "Loss of motivation happens. Start with very small, achievable goals. Sometimes just getting out of bed or taking a shower is enough for today."},
]

# ✅ Predefined Mental Health Responses (keeping your original ones)
mental_health_responses = {
    "anxious": "Take deep breaths. I'm here for you. Try grounding techniques: name 5 things you see, 4 you hear, 3 you feel, 2 you smell, 1 you taste.",
    "stressed": "Step away for a bit, try meditation or deep breathing. Break down your tasks into smaller, manageable pieces.",
    "depressed": "You're not alone. These feelings are valid. Seeking help from a therapist could be helpful. Would you like resources?",
    "lonely": "Connect with someone you trust or engage in an enjoyable activity. Even small social interactions can help.",
    "sad": "It's okay to feel sad. Express your emotions freely. Writing, art, or talking to someone can help process these feelings.",
    "tired": "Ensure you're well-rested and hydrated. Take small breaks throughout the day and listen to your body.",
    "worried": "Try focusing on what you can control. Writing your worries down and creating action plans may help reduce anxiety.",
    "overwhelmed": "When everything feels too much, take it one step at a time. Prioritize essential tasks and ask for help when needed.",
    "hopeless": "These dark feelings won't last forever. Please consider reaching out to a mental health professional or crisis helpline.",
    "panic": "You're experiencing a panic attack, but you're safe. Focus on slow, deep breathing. This will pass."
}

# ✅ Sentiment-Based Response Mapping (keeping your original ones)
sentiment_responses = {
    "POSITIVE": "That's wonderful to hear! It's great that you're feeling positive. What's contributing to these good feelings?",
    "NEGATIVE": "I'm here for you. It's completely okay to feel this way. Would you like to talk about what's troubling you?",
    "NEUTRAL": "Thanks for sharing with me. I'm here to listen. Would you like to talk more about how you're feeling?"
}

# ✅ Fine-tuning Functions
def prepare_training_data(training_data):
    """Prepare training data for DialoGPT fine-tuning"""
    texts = []
    for item in training_data:
        # Format: input + EOS + response + EOS
        text = item["input"] + tokenizer.eos_token + item["response"] + tokenizer.eos_token
        texts.append(text)
    return texts

def tokenize_function(examples):
    """Tokenize the training data"""
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

def fine_tune_model(training_data, epochs=3, learning_rate=5e-5):
    """Fine-tune the DialoGPT model"""
    try:
        # Prepare training data
        texts = prepare_training_data(training_data)
        dataset = Dataset.from_dict({'text': texts})
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./fine_tuned_model",
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            save_steps=10_000,
            save_total_limit=2,
            prediction_loss_only=True,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=100,
            logging_dir='./logs',
            dataloader_pin_memory=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
        )
        
        # Fine-tune
        trainer.train()
        
        # Save model
        trainer.save_model("./fine_tuned_model")
        tokenizer.save_pretrained("./fine_tuned_model")
        
        return True, "Model fine-tuned successfully!"
        
    except Exception as e:
        return False, f"Fine-tuning failed: {str(e)}"

def load_fine_tuned_model():
    """Load the fine-tuned model if it exists"""
    try:
        if os.path.exists("./fine_tuned_model"):
            fine_tuned_model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
            fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
            return fine_tuned_model, fine_tuned_tokenizer
        return None, None
    except:
        return None, None

# ✅ Your original functions (keeping them exactly the same)
def detect_mental_health_issues(user_input):
    user_input = user_input.lower()
    for keyword in mental_health_responses:
        if keyword in user_input:
            return mental_health_responses[keyword]
    return None

def analyze_sentiment(user_input):
    result = sentiment_analyzer(user_input)[0]
    return result['label']

def chatbot_response(user_input, use_fine_tuned=True):
    """Enhanced chatbot response with fine-tuned model option"""
    try:
        # Try to use fine-tuned model first
        current_model = model
        current_tokenizer = tokenizer
        
        if use_fine_tuned and 'fine_tuned_model' in st.session_state and st.session_state.fine_tuned_model:
            current_model = st.session_state.fine_tuned_model
            current_tokenizer = st.session_state.fine_tuned_tokenizer
        
        # Encode input with proper attention mask
        inputs = current_tokenizer.encode(user_input + current_tokenizer.eos_token, 
                                        return_tensors="pt", 
                                        truncation=True, 
                                        max_length=512)
        
        # Create attention mask
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)
        
        with torch.no_grad():
            outputs = current_model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=50,  # Use max_new_tokens instead of max_length
                num_return_sequences=1,
                pad_token_id=current_tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                early_stopping=True
            )
        
        # Decode only the new tokens
        response = current_tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return response.strip() if response.strip() else "I understand you're going through something difficult. Can you tell me more about how you're feeling?"
        
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I'm here to listen. Can you tell me more about what's on your mind?"

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
    """Smart similarity-based response with error handling"""
    try:
        if embedder is None:
            return None
            
        corpus = list(mental_health_responses.keys())
        embeddings = embedder.encode(corpus + [user_input])
        similarities = cosine_similarity([embeddings[-1]], embeddings[:-1])
        best_match_idx = np.argmax(similarities)
        
        if similarities[0][best_match_idx] > 0.5:
            return mental_health_responses[corpus[best_match_idx]]
        return None
    except Exception as e:
        st.warning(f"Similarity matching unavailable: {e}")
        return None

def save_conversation_for_training():
    """Save conversation history for future training"""
    if "chat_history" in st.session_state and len(st.session_state.chat_history) > 0:
        training_file = "user_conversations.json"
        
        # Convert chat history to training format
        conversations = []
        for i in range(0, len(st.session_state.chat_history) - 1, 2):
            if i + 1 < len(st.session_state.chat_history):
                user_msg = st.session_state.chat_history[i][1]
                bot_msg = st.session_state.chat_history[i + 1][1]
                conversations.append({"input": user_msg, "response": bot_msg, "timestamp": str(datetime.now())})
        
        # Save to file
        try:
            if os.path.exists(training_file):
                with open(training_file, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            existing_data.extend(conversations)
            
            with open(training_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            return True
        except:
            return False
    return False

def main():
    st.set_page_config(
        page_title="AI Mental Health Chatbot",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 AI Mental Health Chatbot with Fine-tuning")
    st.write("Chat with an AI-powered assistant that understands your emotional state and learns from interactions.")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "fine_tuned_model" not in st.session_state:
        st.session_state.fine_tuned_model = None
        st.session_state.fine_tuned_tokenizer = None
        # Try to load existing fine-tuned model
        ft_model, ft_tokenizer = load_fine_tuned_model()
        if ft_model:
            st.session_state.fine_tuned_model = ft_model
            st.session_state.fine_tuned_tokenizer = ft_tokenizer
    
    # Sidebar for fine-tuning options
    with st.sidebar:
        st.header("🔧 Model Fine-tuning")
        
        if st.session_state.fine_tuned_model:
            st.success("✅ Fine-tuned model loaded!")
        else:
            st.info("Using base DialoGPT model")
        
        st.subheader("Training Options")
        epochs = st.slider("Training Epochs", 1, 10, 3)
        learning_rate = st.selectbox("Learning Rate", [1e-5, 5e-5, 1e-4], index=1)
        
        if st.button("🚀 Fine-tune with Default Data"):
            with st.spinner("Fine-tuning model... This may take several minutes."):
                success, message = fine_tune_model(mental_health_training_data, epochs, learning_rate)
                if success:
                    st.success(message)
                    # Load the newly fine-tuned model
                    ft_model, ft_tokenizer = load_fine_tuned_model()
                    st.session_state.fine_tuned_model = ft_model
                    st.session_state.fine_tuned_tokenizer = ft_tokenizer
                    st.experimental_rerun()
                else:
                    st.error(message)
        
        st.subheader("Conversation Data")
        if st.button("💾 Save Current Conversation"):
            if save_conversation_for_training():
                st.success("Conversation saved for future training!")
            else:
                st.warning("No conversation to save or save failed.")
        
        # Model selection
        use_fine_tuned = st.checkbox("Use Fine-tuned Model", value=bool(st.session_state.fine_tuned_model))
    
    # Main chat interface
    user_input = st.text_input("You:", "")

    if user_input:
        # Try keyword-based or similarity-based responses first
        response = detect_mental_health_issues(user_input) or smart_response(user_input)

        # If no predefined response found, use DialoGPT (fine-tuned or base)
        if not response:
            response = chatbot_response(user_input, use_fine_tuned)

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
        if sender == "You":
            st.markdown(f"**🧑 {sender}:** {msg}")
        else:
            st.markdown(f"**🤖 {sender}:** {msg}")

    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.experimental_rerun()

    # 🔊 Voice Input (keeping your original implementation)
    st.subheader("🔊 Voice Input")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if uploaded_file:
        text_output = speech_to_text(uploaded_file)
        st.write(f"**Converted Text:** {text_output}")
        
        # Option to use voice input as chat input
        if st.button("Use Voice Input in Chat"):
            if text_output and not text_output.startswith("Error:"):
                st.session_state.voice_input = text_output

    # 🗣️ Text-to-Speech (keeping your original implementation)
    st.subheader("🗣️ Text-to-Speech")
    if st.button("🔊 Speak Last Bot Response"):
        if st.session_state.chat_history:
            last_bot_responses = [msg for sender, msg in st.session_state.chat_history if sender == "Bot"]
            if last_bot_responses:
                last_response = last_bot_responses[-1]
                with st.spinner("Generating speech..."):
                    speech_file = text_to_speech(last_response)
                    st.audio(speech_file, format="audio/mp3")

    # Help section
    with st.expander("ℹ️ How to Use Fine-tuning"):
        st.markdown("""
        **Fine-tuning Options:**
        
        1. **Default Training**: Use the built-in mental health conversation data
        2. **Custom Training**: Have conversations, save them, and retrain the model
        3. **Model Selection**: Choose between base DialoGPT or your fine-tuned version
        
        **Tips:**
        - Start with 3 epochs for initial fine-tuning
        - Save meaningful conversations to improve the model
        - Higher learning rates train faster but may be less stable
        - The fine-tuned model will remember patterns from your training data
        
        **Note:** Fine-tuning requires computational resources and may take time.
        """)

if __name__ == "__main__":
    main()
