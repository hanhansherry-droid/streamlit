import streamlit as st
import pandas as pd
import numpy as np
import pickle #for saving and loading ML models
from sklearn.linear_model import LinearRegression
import time

# =============================================================================
# PART 1: TEXT & INPUTS
# =============================================================================
st.title("Streamlit Demo 222")
st.header("1. Text & Inputs")
# same as 
st.write("### 1. Text & Inputs, but written in Markdown")

# Basic text display
st.write("Hello, this is `st.write()`")
st.markdown("This is **bold** and *italic*")

# Text input - returns string, empty string if nothing entered
name = st.text_input("Your name please here")
if name:
    st.write(f"Hello, {name}!")

# Number input - returns int/float
age = st.number_input("Your age", min_value=0,
                       max_value=120, 
                       value=25)
# st.write(age)
# Slider - returns int/float
score = st.slider("Pick a score", 0, 100, 50)

st.write(f"Age: {age}, Score: {score}")

# splitting the screen into three columns
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Click me"):
        st.write("Clicked!")

with col2:
    checked = st.checkbox("Enable feature")
    if checked:
        st.write("Enabled")

with col3:
    st.metric("Score", 95)

# =============================================================================
# PART 2: TRAIN & SAVE MODEL
# =============================================================================
st.header("2. Train & Save Model")

# Generate dummy data: y = 2.5x + noise
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.6 * X.squeeze() + np.random.randn(100) * 2

df = pd.DataFrame({"x": X.squeeze(), "y": y})
st.dataframe(df.head())  # Interactive table
st.line_chart(df.set_index("x"))  # Quick chart

# Button returns True when clicked, triggers rerun
if st.button("Train & Save Model"):
    model = LinearRegression()
    model.fit(X, y)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success(f"Model saved! Coefficient: {model.coef_[0]:.2f}")

# =============================================================================
# PART 3: LOAD & PREDICT
# =============================================================================
st.header("3. Load & Predict")

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    x_input = st.number_input("Enter X value", value=5.0)
    prediction = model.predict([[x_input]])[0]
    st.metric("Prediction", f"{prediction:.2f}")  # Big number display
except FileNotFoundError:
    st.warning("Train the model first")

# =============================================================================
# PART 4: CHATBOT
# =============================================================================

# Sidebar for settings
st.sidebar.header("Chatbot Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
model = st.sidebar.selectbox("Model", ["gpt-4", "claude-3"])

# Main area keeps the chat
st.header("4. Chatbot")

# Session state is a dictionary that saves the data that persists across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):  # "user" or "assistant"
        st.write(msg["content"])

# Chat input - returns None until user submits
if prompt := st.chat_input("Say something"):

    # above is called walrus operator
    # it is the same as
    # prompt = st.chat_input("Say something")
    #   if prompt:
    #       do_something(prompt)

    # 
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        response = f"You said: {prompt}"  # Replace with LLM call
        
        # Generator for streaming effect
        def stream():
            for word in response.split():
                yield word + " "
                time.sleep(0.5) # control how slow or fast you want the streaming effect to be
        
        # write_stream consumes generator, returns full text
        full_response = st.write_stream(stream())
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})