from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
#from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from docx import Document
import pandas as pd
from pathlib import Path

load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings()
# llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

class DocumentStorage():
    
    def set_document_vectorstore(text_chunks, doc_type, doc_name):
        metadatas = [{"type": doc_type, "name": doc_name} for _ in range(len(text_chunks))]
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings, metadatas=metadatas)
        vectorstore.save_local("faiss_index")

        return vectorstore
    
    def set_llama_index():
        documents = SimpleDirectoryReader('docs').load_data()
        index = VectorStoreIndex.from_documents(documents)

        # # save index to disk
        # index.set_index_id("vector_index")
        # index.storage_context.persist("./storage")
        # # rebuild storage context
        # storage_context = StorageContext.from_defaults(persist_dir="storage")
        # # load index
        # index = load_index_from_storage(storage_context, index_id="vector_index")
        return index
        
    

class DocumentProcessing():

    def get_text_file_content(text_files):
        text = ""
        for text_file in text_files:
            text += text_file.read().decode("utf-8")
        return text

    def get_csv_file_content(csv_files):
        text = ""
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            text += df.to_string()
        return text

    def get_pdf_file_content(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_docx_file_content(docx_docs):
        text = ""
        for docx in docx_docs:
            doc = Document(docx)
            for para in doc.paragraphs:
                text += para.text
        return text

    def get_text_chunks(text ,chunk_size, chunk_overlap):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks