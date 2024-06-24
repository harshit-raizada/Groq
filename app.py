# Importing libraries

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load the environment variables

load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

# Initialize session state in Streamlit & data ingestion, chunks & embeddings

if 'vector' not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader('https://www.hcltech.com/trends-and-insights/decoding-genai-future-artificial-intelligence')
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_document = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_document, st.session_state.embeddings)

# Set up Streamlit UI

st.set_page_config(page_title = "ChatGroq with Mixtral", page_icon = ":robot_face:", layout = "wide")
st.title('ChatGroq using Mixtral')

# Sidebar for additional information

st.sidebar.header('About this App')
st.sidebar.write("""
This application uses Mixtral, a powerful language model, to provide accurate answers based on the given context. 
The data is sourced from [HCLTech](https://www.hcltech.com/trends-and-insights/decoding-genai-future-artificial-intelligence).
""")

st.sidebar.header('Instructions')
st.sidebar.write("""
1. Enter your question in the text box below.
2. The model will retrieve relevant information and provide an accurate response.
3. Make sure your question is clear and concise for the best results.
""")

# LLM setup

llm = ChatGroq(model_name = 'mixtral-8x7b-32768', groq_api_key = groq_api_key)

# Designing prompt

prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Creating Retriever

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)

# Input prompt

st.subheader('Ask a Question')
user_prompt = st.text_input('Input your prompt here')

if user_prompt:
    with st.spinner('Retrieving and generating response...'):
        response = retriever_chain.invoke({'input': user_prompt})
        st.write(response['answer'])