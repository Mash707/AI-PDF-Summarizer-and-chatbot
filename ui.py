import streamlit as st
import tempfile
from main import Agent
import os
import time

st.set_page_config(page_title="Research Chatbot", layout="wide")

api_key = "f1z5Dvqlmskwgyf9sBshN2f1IVfQ6Lcn6uDbbiRb"

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_doc" not in st.session_state:
    st.session_state.processed_doc = None

with st.sidebar:
    st.caption("Upload research paper (PDF)")
    uploaded_file = st.file_uploader("Choose file", type="pdf")

if uploaded_file and api_key:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    if "agent" not in st.session_state:
        with st.spinner("Processing document..."):
            st.session_state.agent = Agent(api_key)
            st.session_state.agent.load_paper(tmp_path)
            st.session_state.agent.document_embeddings(tmp_path)
            st.session_state.processed_doc = uploaded_file.name

st.title("Research Paper Chatbot")
st.subheader(f"Active Document: {st.session_state.processed_doc or 'None'}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the document..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Analyzing..."):
            raw_response = st.session_state.agent.rag(tmp_path, prompt)
            
            for word in raw_response.split():
                full_response += word + " "
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)  
            
            response_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

