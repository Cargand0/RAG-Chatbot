import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import pdfplumber
import csv
from datetime import datetime
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import logging
import shutil
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.chains.question_answering import load_qa_chain
from os.path import expanduser
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Blip = image processor
model_dir = r"D:\LLM\Blip"
blip_model = BlipForConditionalGeneration.from_pretrained(model_dir)
blip_processor = BlipProcessor.from_pretrained(model_dir)

def get_image_description(image_path):
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        return "Image file not found."

    try:
        image = Image.open(image_path).convert("RGB")
        inputs = blip_processor(images=image, return_tensors="pt")
        out = blip_model.generate(
            **inputs,
            max_length=200,
            num_beams=10,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
            early_stopping=True
        )
        logging.info(f"Raw model output: {out}")
        description = blip_processor.decode(out[0], skip_special_tokens=True)
        logging.info(f"Generated description: {description}")
        return description
    except Exception as e:
        logging.error(f"Failed to process image {image_path}: {e}")
        return "Failed to describe the image."

if 'progress' not in st.session_state:
    st.session_state.progress = 0

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def split_into_chunks(text, chunk_size, chunk_overlap):
    num_chunks = max(1, (len(text) - chunk_overlap) // (chunk_size - chunk_overlap))
    chunks = []
    for i in range(num_chunks):
        start = i * (chunk_size - chunk_overlap)
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
    return chunks

def extract_text_from_file(filename):
    try:
        if filename.endswith('.pdf'):
            return extract_text_from_pdf(filename)
        elif filename.endswith('.txt'):
            return extract_text_from_txt(filename)
        elif filename.endswith('.csv'):
            return extract_text_from_csv(filename)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    except Exception as e:
        logging.error(f"Error processing file {filename}: {e}")
        return []

def extract_text_from_pdf(filename):
    try:
        with pdfplumber.open(filename) as pdf:
            text = []
            for page in pdf.pages:
                text.extend(page.extract_text().split('\n\n'))
            return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF {filename}: {e}")
        return []

def extract_text_from_txt(filename):
    try:
        with open(filename, 'r') as file:
            text = file.read().split('\n\n')
        return text
    except Exception as e:
        logging.error(f"Error extracting text from TXT {filename}: {e}")
        return []

def extract_text_from_csv(filename):
    try:
        with open(filename, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            text = [" ".join(row) for row in reader if row]
        return text
    except Exception as e:
        logging.error(f"Error extracting text from CSV {filename}: {e}")
        return []

def process_files(files):
    documents = []
    metadata = []
    filenames = []
    for file in files:
        text_blocks = extract_text_from_file(file)
        current_timestamp = datetime.now().isoformat()
        for text in text_blocks:
            if len(text) >= 750:
                chunks = split_into_chunks(text, chunk_size=750, chunk_overlap=150)
            else:
                chunks = [text]
            documents.extend(chunks)
            filenames.extend([file] * len(chunks))
            metadata.extend([{"filename": file, "timestamp": current_timestamp, "type": "text"} for _ in chunks])
            logging.info(f"Processed {len(chunks)} chunks from {file}")
    return documents, metadata, filenames

def store_documents(documents, metadata, filenames, collection):
    try:
        ids = [f"{os.path.basename(filenames[i]).replace('.', '_')}_{i}" for i in range(len(documents))]
        for doc, meta, doc_id in zip(documents, metadata, ids):
            collection.add_texts([doc], metadatas=[meta], ids=[doc_id])
        logging.info(f"Stored documents with IDs: {ids}")
    except Exception as e:
        logging.error(f"Error storing documents: {e}")

def store_images_in_chroma(images, collection, project_dir="images"):
    metadata = []
    image_ids = []
    documents = []

    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    for image_path in images:
        try:
            description = get_image_description(image_path)
            documents.append(description)
            image_id = f"id_{os.path.basename(image_path).replace('.', '_')}"
            meta = {"filename": image_path, "timestamp": datetime.now().isoformat(), "type": "image"}
            metadata.append(meta)
            image_ids.append(image_id)
            
            shutil.copy(image_path, os.path.join(project_dir, os.path.basename(image_path)))
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            continue

    try:
        for doc, meta, doc_id in zip(documents, metadata, image_ids):
            collection.add_texts([doc], metadatas=[meta], ids=[doc_id])
        logging.info(f"Stored images with IDs: {image_ids} in collection.")
    except Exception as e:
        logging.error(f"Failed to add images: {e}")

prompt_template = """
As an AI assistant, you provide answers based on the given source documents, ensuring accuracy and briefness.

You always follow these guidelines:
- Always answer in English language
- Only answer from the uploaded source documents
- Do not give additional information outside from the source documents
- If the answer isn't available within the source documents, respond that it is not available
- Otherwise, answer to your best capability, only referring to the source documents provided
- Only use examples if explicitly requested
- Do not introduce examples outside of the source documents
{context}

Question: {question}
Helpful Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

system_template = """
As an AI assistant, you provide answers based on the given source documents, ensuring accuracy and briefness.

You always follow these guidelines:
- Always answer in English language
- Only answer from the uploaded source documents
- Do not give additional information outside from the source documents
- If the answer isn't available within the source documents, respond that it is not available
- Otherwise, answer to your best capability, only referring to the source documents provided
- Only use examples if explicitly requested
- Do not introduce examples outside of the source documents
{context}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)

model_path = expanduser("llama-2-13b-chat.Q3_K_M.gguf")
default_ef = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
collection = Chroma(
    collection_name="documents",
    embedding_function=default_ef,
    persist_directory="joeDb",
)
docsearch = collection.as_retriever()
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = 30
n_batch = 512
n_ctx = 2048
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=n_ctx,
)
model = Llama2Chat(llm=llm, verbose=True, callback_manager=callback_manager)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = load_qa_chain(model, chain_type="stuff", callback_manager=callback_manager)

def rerank_documents(docs_with_score, query):
    adjusted_docs_with_scores = []
    for doc, score in docs_with_score:
        freshness = compute_freshness(doc.metadata.get('timestamp', ''))
        keyword_alignment = compute_keyword_relevance(doc.page_content, query)
        adjusted_score = score * 0.5 + freshness * 0.3 + keyword_alignment * 0.2
        adjusted_docs_with_scores.append((doc, adjusted_score))

    reranked_docs_with_score = sorted(adjusted_docs_with_scores, key=lambda x: x[1], reverse=True)
    return reranked_docs_with_score

def compute_freshness(timestamp):
    try:
        current_time = datetime.now()
        document_time = datetime.fromisoformat(timestamp)
        time_diff = (current_time - document_time).days
        return max(1 - (time_diff / 365), 0)
    except Exception as e:
        logging.error(f"Error computing freshness: {e}")
        return 0.5

def compute_keyword_relevance(content, query):
    keywords = query.split()
    content_words = content.split()
    common_words = set(keywords) & set(content_words)
    return len(common_words) / len(keywords) if keywords else 0

st.title("Chat with Documents")

if not os.path.exists("uploads"):
    os.makedirs("uploads")

st.header("Upload Documents")
uploaded_files = st.file_uploader("Choose files", type=["pdf", "txt", "csv"], accept_multiple_files=True)
if uploaded_files:
    filenames = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        filenames.append(file_path)
    documents, metadata, original_filenames = process_files(filenames)
    store_documents(documents, metadata, original_filenames, collection)
    st.success("Documents processed and stored successfully.")

st.header("Upload Images")
uploaded_images = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_images:
    image_paths = []
    for uploaded_image in uploaded_images:
        file_path = os.path.join("uploads", uploaded_image.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        image_paths.append(file_path)
    store_images_in_chroma(image_paths, collection)
    st.success("Images processed and stored successfully.")

st.header("Chatbot:")
user_input = st.text_input("Type your question here:", key="user_input", label_visibility="collapsed")

def handle_user_input(user_input):
    if user_input:
        st.session_state.chat_history.append(("You", user_input))
        try:
            if user_input.lower().startswith('describe '):
                image_path = user_input[len('describe '):].strip()
                description = get_image_description(image_path)
                st.session_state.chat_history.append(("Chatbot", description))
            elif user_input.lower().startswith('display '):
                image_name = user_input[len('display '):].strip()
                project_dir = "images"
                image_path = os.path.join(project_dir, image_name)
                if os.path.exists(image_path):
                    st.session_state.chat_history.append(("Chatbot", f"Displaying image {image_name}"))
                    st.image(image_path)  
                else:
                    st.session_state.chat_history.append(("Chatbot", "Image file not found."))
            else:
                st.write("Chatbot is generating the answers...")
                placeholder = st.empty()
                try:
                    docs_with_score = docsearch.vectorstore.similarity_search_with_relevance_scores(user_input, k=100, score_threshold=0.4)
                    if not docs_with_score:
                        response = chain.invoke({"question": user_input, "input_documents": [], "metadata": []})
                    else:
                        reranked_docs_with_score = rerank_documents(docs_with_score, user_input)
                        docs = [doc for doc, _ in reranked_docs_with_score]
                        metadata = [doc.metadata['filename'] for doc, _ in reranked_docs_with_score]

                        response = chain.invoke({"question": user_input, "input_documents": docs, "metadata": metadata})
                    
                    placeholder.write(f"Chatbot: {response['output_text']}")
                    st.session_state.chat_history.append(("Chatbot", response['output_text']))
                except Exception as e:
                    placeholder.write("Chatbot: Failed to retrieve answer. Please try again.")
                    logging.error(f"Failed to retrieve answer: {e}")
                    st.session_state.chat_history.append(("Chatbot", "Failed to retrieve answer. Please try again."))
        except MemoryError as e:
            st.error("The operation failed due to insufficient RAM. Please close some applications or increase the server capacity, and try again.")
            logging.error(f"MemoryError encountered: {str(e)}")
            st.session_state.chat_history.append(("Chatbot", "Failed to retrieve answer due to insufficient RAM. Retrying..."))
            handle_user_input(user_input)
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            st.session_state.chat_history.append(("Chatbot", "An error occurred. Please try again."))

handle_user_input(user_input)

for role, message in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"<div style='text-align: right;'><b>{role}:</b> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left;'><b>{role}:</b> {message}</div>", unsafe_allow_html=True)
