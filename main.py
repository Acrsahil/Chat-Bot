import streamlit as st
import datetime
import dateparser
import faiss
import numpy as np
import os
import re
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

# Set Hugging Face API Token (set this in your environment variables)
load_dotenv()  # Load environment variables from .env

hf_api_token = os.getenv("HF_API_TOKEN")  # Retrieve the token safely

if hf_api_token is None:
    raise ValueError("HF_API_TOKEN is not set in .env")

# Initialize Hugging Face LLM
llm = HuggingFaceHub(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    model_kwargs={"temperature": 0.6, "max_length": 64},
    huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
)

vector_store = None

st.set_page_config(page_title="📄 AI Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 AI Chatbot with Q&A & Appointment Booking")

# Sidebar for file upload
st.sidebar.header("📄 Upload Documents")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        # Save the uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        # Load the PDF using the saved file path
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load_and_split()

        # Initialize HuggingFaceEmbeddings without passing api_key directly
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = FAISS.from_documents(pages, embeddings)

    st.sidebar.success("Document uploaded & processed! ✅")


# Function to extract valid date
def extract_date(user_input):
    parsed_date = dateparser.parse(user_input)
    if parsed_date:
        return parsed_date.strftime("%Y-%m-%d")
    return None


# Function to validate email & phone
def validate_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)


def validate_phone(phone):
    return re.match(r"^\+?\d{10,15}$", phone)


# Chat interface
st.subheader("💬 Chat with AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    if "call me" in user_input.lower():
        with st.form("appointment_form"):
            name = st.text_input("📛 Name")
            email = st.text_input("📧 Email")
            phone = st.text_input("📞 Phone Number")
            date_query = st.text_input("📅 Preferred Date (e.g., Next Monday)")
            submitted = st.form_submit_button("Submit")

            if submitted:
                if not validate_email(email):
                    st.error("Invalid email format! ❌")
                elif not validate_phone(phone):
                    st.error("Invalid phone number! ❌")
                else:
                    appointment_date = (
                        extract_date(date_query) if date_query else "Not provided"
                    )
                    st.success(
                        f"✅ Appointment booked for {name} on {appointment_date}. We will contact you at {phone}."
                    )
    else:
        if vector_store:
            # Perform similarity search to find the most relevant document
            results = vector_store.similarity_search(user_input, k=1)

            # Extract the content of the most relevant document
            context = (
                results[0].page_content  # Extract the page content from the results
                if results
                else "No relevant info found in the document."
            )

            # Form the input for the LLM: combine user input and document context
            input_for_llm = f"User question: {user_input}\nRelevant information from the document:\n{context}"

        else:
            input_for_llm = f"User question: {user_input}\nSorry, I don't have any documents to refer to."

        # Call Hugging Face model with the combined input
        bot_response = llm(input_for_llm)

        with st.chat_message("assistant"):
            st.markdown(bot_response)

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
