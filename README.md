# 💬 Neal's Ice Machine Technician using RAG Methodology

This Streamlit application leverages OpenAI's GPT-3.5 and FAISS to provide a chatbot that can answer questions about the Manitowoc Indigo NXT commercial ice machine. The bot uses embeddings to search for relevant information in a pre-built FAISS index and generates accurate responses based on the documents.

https://ice-machine-technician-iqrapprnxetp5khifu73wwh.streamlit.app/

## Features

- **User Authentication**: Users can input their OpenAI API key to use the application.
- **Document Search**: Uses FAISS to search through a pre-built index of documents.
- **Chat Interface**: Streamlit's chat interface allows users to interact with the bot and get answers to their questions.

## Requirements

- Python 3.7 or higher
- `faiss-cpu`
- `openai`
- `streamlit`
- Other dependencies listed in `requirements.txt`

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Add the FAISS index and metadata files**:
    Ensure `faiss_index.bin` and `metadata_store.json` are in the same directory as `app.py`.

## Usage

1. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2. **Open the app**:
    Open the URL provided by Streamlit in your browser.

3. **Enter your OpenAI API Key**:
    Input your OpenAI API key in the sidebar to use the application. You can get an API key from [OpenAI](https://platform.openai.com/account/api-keys).

4. **Interact with the bot**:
    Type your questions into the chat input at the bottom of the interface. The bot will respond with relevant information based on the documents indexed by FAISS.

## Code Overview

### Main Script (`app.py`)

```python
import openai
import streamlit as st
import faiss
import json
import numpy as np

# Load the FAISS index and metadata
index = faiss.read_index('faiss_index.bin')
with open('metadata_store.json', 'r') as f:
    metadata_store = json.load(f)

# Sidebar for API key input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

st.title("💬 Neal's Ice Machine Technician")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask me a question about the Manitowoc Indigo NXT commercial ice machine."}]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Function to get embedding from OpenAI
def get_embedding(text, api_key):
    openai.api_key = api_key
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return np.array(response['data'][0]['embedding'])

# Function to search FAISS index
def search_faiss(query, api_key, k=5):
    query_vector = get_embedding(query, api_key)
    distances, indices = index.search(np.array([query_vector]), k)
    results = []
    for i in range(k):
        idx = indices[0][i]
        metadata = metadata_store[str(idx)]
        results.append({
            'id': idx,
            'distance': distances[0][i],
            'metadata': metadata
        })
    return results

# Function to generate response using OpenAI
def generate_response(query, documents, api_key):
    openai.api_key = api_key
    context = ""
    for doc in documents:
        context += f"\nPage {doc['metadata']['page_number']} - {doc['metadata']['type']}:\n{doc['metadata'].get('text', '')}\n"

    messages = [
        {"role": "system", "content": "You are a helpful assistant that uses provided documents to answer the user's questions as accurately as possible."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": f"Here are some relevant documents:\n{context}\nUse the information in these documents to answer the user's question as accurately as possible. If the specific information is not found, provide guidance based on the document context."}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7
    )
    
    return response.choices[0].message['content'].strip()

# User input
if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue. No keys will be stored.")
        st.stop()

    # Display user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Search FAISS and generate response
    try:
        search_results = search_faiss(prompt, openai_api_key)
        bot_response = generate_response(prompt, search_results, openai_api_key)
    except Exception as e:
        bot_response = f"Request failed: {e}"
    
    # Display bot response
    msg = {"role": "assistant", "content": bot_response}
    st.session_state["messages"].append(msg)
    st.chat_message("assistant").write(bot_response)
