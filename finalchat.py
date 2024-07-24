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
