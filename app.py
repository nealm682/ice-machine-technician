from flask import Flask, request, jsonify
import faiss
import json
import numpy as np
import openai

app = Flask(__name__)

# Load the FAISS index and metadata
index = faiss.read_index('faiss_index.bin')
with open('metadata_store.json', 'r') as f:
    metadata_store = json.load(f)

def get_embedding(text, api_key):
    openai.api_key = api_key
    response = openai.Embedding.create(model="text-embedding-ada-002", input=[text])
    return np.array(response['data'][0]['embedding'])

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
    
    return response.choices[0]['message']['content'].strip()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'query' not in data or 'api_key' not in data:
        return jsonify({'error': 'No query or API key provided'}), 400
    
    query = data['query']
    api_key = data['api_key']
    print(f"Received query: {query}")  # Log received query
    try:
        search_results = search_faiss(query, api_key)
        response = generate_response(query, search_results, api_key)
        print(f"Generated response: {response}")  # Log generated response
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error processing request: {e}")  # Log any errors
        return jsonify({'error': 'Error processing request'}), 500

if __name__ == '__main__':
    app.run(debug=True)
