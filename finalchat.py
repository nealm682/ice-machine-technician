import openai
import streamlit as st
import requests

# Sidebar for API key input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    
st.title("💬 Neal's Ice Machine Technician")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask me a question about the Manitowoc Indigo NXT commercial ice machine."}]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue. No keys will be stored.")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Send user prompt to backend and get response
    try:
        response = requests.post('http://127.0.0.1:5000/chat', json={'query': prompt, 'api_key': openai_api_key})
        response.raise_for_status()  # Raise an error for bad status codes
        response_data = response.json()
        bot_response = response_data.get('response', 'No response from server.')
    except requests.exceptions.RequestException as e:
        bot_response = f"Request failed: {e}"
    
    # Display bot response
    msg = {"role": "assistant", "content": bot_response}
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(bot_response)
