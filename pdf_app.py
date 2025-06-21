import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import google.generativeai as genai
import tempfile
import uuid
import hashlib # For creating content hashes

# --- Configuration Functions ---

def load_env_and_configure_genai():
    """Loads environment variables and configures the Google Generative AI API."""
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("🚨 Error: Please set your GOOGLE_API_KEY environment variable in a `.env` file or Streamlit secrets.")
        st.stop()
    else:
        genai.configure(api_key=google_api_key)

def get_text_chunks_with_metadata(pdf_docs, chunk_size, chunk_overlap):
    """
    Reads text from multiple uploaded PDF files, splits the text into chunks, and adds metadata.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []

    for pdf_doc in pdf_docs:
        temp_file_id = str(uuid.uuid4())
        temp_filepath = os.path.join(tempfile.gettempdir(), f"{temp_file_id}_{pdf_doc.name}")

        try:
            # Ensure the file pointer is at the beginning before reading for PDFReader
            pdf_doc.seek(0)
            with open(temp_filepath, "wb") as f:
                f.write(pdf_doc.read())
            
            reader = PdfReader(temp_filepath)
            source_name = os.path.basename(pdf_doc.name).replace(".pdf", "")
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    chunks = splitter.split_text(text)
                    for j, chunk in enumerate(chunks):
                        all_chunks.append({
                            "text": chunk,
                            "metadata": {"source": source_name, "page": i + 1, "chunk_id": j}
                        })
        except Exception as e:
            st.error(f"⚠️ Error processing PDF '{pdf_doc.name}': {e}")
        finally:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
        # Reset file pointer for potential future reads (e.g., if passed to another function)
        pdf_doc.seek(0)
    return all_chunks

# This function must be defined BEFORE main()
@st.cache_resource(show_spinner="Building knowledge base... this might take a moment!")
def get_vector_store(chunks_data):
    """
    Generates embeddings for given text chunks and stores them in a FAISS vector database.
    """
    texts = [c["text"] for c in chunks_data]
    metadatas = [c["metadata"] for c in chunks_data]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    return db

# This function must be defined BEFORE main()
@st.cache_data(show_spinner="Summarizing documents...")
def summarize_pdf(_content_hash, file_name, temp, _uploaded_file_bytes):
    """
    Reads the entire text from a PDF and uses an LLM to generate a summary.
    _content_hash: A unique hash of the PDF's content, used for caching.
    file_name: The name of the PDF file.
    temp: Model temperature.
    _uploaded_file_bytes: The actual bytes of the PDF, marked unhashable as it's large.
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temp)
    
    temp_file_id = str(uuid.uuid4())
    tmp_file_path = os.path.join(tempfile.gettempdir(), f"{temp_file_id}_{file_name}")
    
    try:
        with open(tmp_file_path, "wb") as tmp_file:
            tmp_file.write(_uploaded_file_bytes)
        
        reader = PdfReader(tmp_file_path)
        full_text = ""
        for page in reader.pages:
            text_on_page = page.extract_text()
            if text_on_page:
                full_text += text_on_page + "\n" # Add newline for better separation

        prompt = PromptTemplate(
            input_variables=["document"],
            template="Summarize this document in less than 100 words:\n{document}"
        )
        return model.invoke(prompt.format(document=full_text)).content
    except Exception as e:
        st.warning(f"Could not summarize {file_name}: {e}")
        return "Summary not available."
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# This function must be defined BEFORE main()
@st.cache_resource(show_spinner="Initializing conversation model...")
def load_qa_chain(_vector_store_instance, temp):
    """
    Sets up and returns a LangChain ConversationalRetrievalChain for question answering.
    """
    retriever = _vector_store_instance.as_retriever(search_kwargs={"k": 32})
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temp)

    qa_prompt = PromptTemplate(
        template="""
        You are an AI assistant for question-answering over documents.
        Use the retrieved context to answer comprehensively.
        If a question covers multiple entities, include all of them.
        If data is missing, say: 'I cannot find the answer to this question in the provided documents.'

        Chat History:
        {chat_history}

        Context:
        {context}

        Question: {question}
        Answer:
        """,
        input_variables=["question", "context", "chat_history"]
    )

    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

# This function must be defined BEFORE main()
@st.cache_data(show_spinner="Highlighting relevant sources...")
def highlight_relevant_sources_full_chunk(answer, _source_documents_data, temp):
    """
    Uses an LLM to identify and highlight relevant sentences within the full text of
    the source document chunks that were used to generate a given answer.
    _source_documents_data: A list of tuples, where each tuple is
                            (page_content, source, page, chunk_id).
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temp)

    formatted_chunks = "\n\n".join(
        f"[Source: {source} | Page: {page} | Chunk ID: {chunk_id}]\n{page_content}"
        for page_content, source, page, chunk_id in _source_documents_data
    )

    prompt = PromptTemplate(
        input_variables=["answer", "context"],
        template="""
You are given an answer and a set of document chunks from PDFs.

Task:
- For each chunk that supports the answer:
  - Return the full chunk.
  - Highlight the relevant sentences using **double asterisks**.
  - Prepend the source info like:
    ════════════════════════════════════════════════════════════════
    📄 Source: <source> | Page: <page>  | Chunk ID: <chunk_id>
    ════════════════════════════════════════════════════════════════

- Skip chunks that are not relevant.
- Add a line: ────────────────────────────────────────────── after each source block.

Answer:
{answer}

Document Chunks:
{context}

Output format:

════════════════════════════════════════════════════════════════
📄 Source: <source> | Page: <page> | Chunk ID: <chunk_id>
════════════════════════════════════════════════════════════════
<Full Chunk with **highlighted** text>

────────────────────────────────────────────────────────────────
"""
    )
    response = model.invoke(prompt.format(answer=answer, context=formatted_chunks))
    return response.content

# --- Streamlit Application ---
def main():
    st.set_page_config(page_title="Chat with PDF 💬", layout="wide", page_icon=":books:")
    st.title("📄 Chat with your PDFs using Gemini + LangChain")
    st.markdown("Upload multiple PDF documents, process them, and then ask questions. The AI will summarize each document and answer your questions based on the content of all uploaded PDFs.")

    # Initialize session state variables
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "pdf_summaries" not in st.session_state:
        st.session_state.pdf_summaries = {}
    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = False
    if "uploaded_files_hashes" not in st.session_state:
        st.session_state.uploaded_files_hashes = {}
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""


    # --- Load Environment Variables and Configure Google API ---
    load_env_and_configure_genai()

    # --- Sidebar for Settings and PDF Upload ---
    with st.sidebar:
        st.header("⚙️ Settings")
        
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, help="The maximum size of text chunks processed by the model.", key="chunk_size_slider")
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 300, help="The number of characters that overlap between consecutive text chunks.", key="chunk_overlap_slider")
        temperature = st.slider("Model Temperature", 0.0, 1.0, 0.3, help="Controls the randomness of the model's output. Lower is more deterministic.", key="temperature_slider")

        pdf_docs = st.file_uploader(
            "Upload PDF(s)",
            accept_multiple_files=True,
            type=["pdf"],
            key="pdf_uploader_widget",
            help="Upload one or more PDF files to chat with."
        )

        # Process PDFs button
        if st.button("🚀 Process PDFs", key="process_pdfs_button"):
            if pdf_docs:
                st.session_state.vector_store_ready = False
                st.session_state.pdf_summaries = {}
                st.session_state.chat_history = []
                st.session_state.last_response = None
                st.session_state.uploaded_files_hashes = {}

                with st.spinner("Processing documents..."):
                    documents_to_process_for_chunks = []
                    for pdf_doc in pdf_docs:
                        pdf_doc.seek(0)
                        content_bytes = pdf_doc.read()
                        
                        content_hash = hashlib.md5(content_bytes).hexdigest()
                        
                        st.session_state.uploaded_files_hashes[pdf_doc.name] = content_hash
                        
                        summary = summarize_pdf(content_hash, pdf_doc.name, temperature, content_bytes)
                        st.session_state.pdf_summaries[pdf_doc.name] = summary
                        
                        pdf_doc.seek(0)
                        documents_to_process_for_chunks.append(pdf_doc)
                    
                    all_chunks = get_text_chunks_with_metadata(documents_to_process_for_chunks, chunk_size, chunk_overlap)
                    
                    if all_chunks:
                        st.session_state.vector_store = get_vector_store(all_chunks)
                        st.session_state.qa_chain = load_qa_chain(st.session_state.vector_store, temperature)
                        st.session_state.vector_store_ready = True
                        st.success("Documents processed and ready for chat!")
                    else:
                        st.warning("No text could be extracted from the uploaded PDFs. Please check the files.")
                        st.session_state.vector_store_ready = False
            else:
                st.warning("Please upload at least one PDF document to process.")

    # --- Document Summary Section ---
    st.markdown("---")
    st.subheader("📚 Document Summaries")
    if st.session_state.pdf_summaries:
        for pdf_name, summary_text in st.session_state.pdf_summaries.items():
            with st.expander(f"**Summary for:** {pdf_name}"):
                st.markdown(f'''
                    <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; color:#333333; font-size: 0.95em;">
                    {summary_text}
                    </div>
                ''', unsafe_allow_html=True)
    else:
        st.info("Upload PDFs and click 'Process' to see their summaries here.")

    # --- Chat Input Section ---
    st.markdown("---")
    
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input(
            "Ask a question about your documents:",
            key="user_question_input_box",
            placeholder="Type question to get response",
            value=st.session_state.user_question
        )
        submitted = st.form_submit_button("Submit")
    
    if submitted and user_question.strip() != "":
        st.session_state.user_question = ""
        if not st.session_state.vector_store_ready or st.session_state.qa_chain is None:
            st.warning("Please process your PDFs first in the sidebar.")
        else:
            with st.spinner("Thinking..."):
                try:
                    formatted_chat_history = []
                    for turn in st.session_state.chat_history:
                        formatted_chat_history.append(HumanMessage(content=turn["question"]))
                        formatted_chat_history.append(AIMessage(content=turn["answer"]))

                    response = st.session_state.qa_chain.invoke(
                        {"question": user_question, "chat_history": formatted_chat_history}
                    )
                    
                    st.session_state.last_response = response
                    st.session_state.chat_history.append({"question": user_question, "answer": response["answer"]})

                except Exception as e:
                    st.error(f"An error occurred while answering your question: {e}")
                    st.session_state.last_response = None
    elif submitted and user_question.strip() == "":
        st.warning("Please type a question.")

    # --- Answer Section ---
    st.markdown("### 💬 Answer:")
    if st.session_state.last_response:
        st.info(st.session_state.last_response["answer"])
    else:
        st.markdown("The model's answer will appear here.")

    # --- Sources Section ---
    st.markdown("---")
    st.markdown("### 🔍 Sources:")
    st.markdown(
        """
        <style>
        .source-container {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            margin-bottom: 10px;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.last_response and st.session_state.last_response.get("source_documents"):
        hashable_source_docs = [
            (doc.page_content, doc.metadata.get('source', 'N/A'), doc.metadata.get('page', 'N/A'), doc.metadata.get('chunk_id', 'N/A'))
            for doc in st.session_state.last_response["source_documents"]
        ]
        highlighted_full_chunks = highlight_relevant_sources_full_chunk(
            st.session_state.last_response["answer"],
            hashable_source_docs,
            temperature
        )
        st.markdown(f'<div class="source-container">{highlighted_full_chunks}</div>', unsafe_allow_html=True)
    else:
        st.info("💡 This section will show snippets from the original documents that were used to generate the answer.")

    # --- Conversation History Section ---
    st.markdown("---")
    st.subheader("🧠 Conversation History")
    if st.session_state.chat_history:
        for i, chat_turn in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"**Question {len(st.session_state.chat_history) - i}:** {chat_turn['question'][:70]}{'...' if len(chat_turn['question']) > 70 else ''}"):
                st.markdown(f"**You:** {chat_turn['question']}")
                st.markdown(f"**Bot:** {chat_turn['answer']}")
                st.markdown("---")
    else:
        st.info("No conversation history yet. Ask a question after processing your PDFs!")


if __name__ == "__main__":
    main()