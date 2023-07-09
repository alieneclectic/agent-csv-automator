import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template
from utils import DocumentStorage, DocumentProcessing
from custom_tools import Custom_Tools
from agent import Agent
import pandas as pd
import openai


def handle_userinput(user_question):
    if user_question:

        agent = Agent.initialize_conversational_agent()
        st.session_state.conversation = agent

        # Truncate or summarize the conversation history if it's too long
        if len(st.session_state.chat_history) > 1000:
            st.session_state.chat_history = st.session_state.chat_history[-1000:]

        try:
            #call the OpenAI API
            response = st.session_state.conversation({'input': user_question})
        except openai.error.InvalidRequestError as e:
            # Handle the error
            st.error(f"Error: {str(e)}")

        # Append the response to the existing chat history
        if 'chat_history' in response:
            #st.session_state.chat_history.extend(response['chat_history'])
            st.session_state.chat_history = response['chat_history']

        # Reverse the chat history in pairs
        reversed_chat_history = list(reversed([st.session_state.chat_history[i:i+2] for i in range(0, len(st.session_state.chat_history), 2)]))
        flattened_chat_history = [item for sublist in reversed_chat_history for item in sublist]
        
        for i, message in enumerate(flattened_chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)



def handle_document_upload():
    with st.sidebar:
        st.subheader("Documents")
        uploaded_docs = st.file_uploader(
            "Upload your documents here and click on 'Upload'", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        if uploaded_docs:
            st.session_state.uploaded_docs = uploaded_docs
        if st.button("Upload Documents", key="button1"):
            if not st.session_state.uploaded_docs:
                st.warning("Please upload a document first before processing.")
            else:
                with st.spinner("Processing"):
                    raw_text = ""
                    for doc in st.session_state.uploaded_docs:
                        if doc.type == "application/pdf":
                            raw_text += DocumentProcessing.get_pdf_file_content([doc])
                        elif doc.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            raw_text += DocumentProcessing.get_docx_file_content([doc])
                        elif doc.type == "text/plain":
                            raw_text += DocumentProcessing.get_text_file_content([doc])
                        elif doc.type == "text/csv":
                            raw_text += DocumentProcessing.get_csv_file_content([doc])
                            
                    text_chunks = DocumentProcessing.get_text_chunks(raw_text, chunk_size=1000, chunk_overlap=100)
                    # create vector store
                    vectorstore = DocumentStorage.set_document_vectorstore(text_chunks, doc.type, doc.name)
                          
                    
def handle_csv_upload():
    with st.sidebar:
        st.subheader("CSVs")
        uploaded_csv = st.file_uploader(
            "Upload your CSVs here and click on 'Upload'", type=['csv'], accept_multiple_files=True)
        if uploaded_csv:
            st.session_state.uploaded_csv = uploaded_csv
        if st.button("Upload & CSV", key="button2"):
            if not st.session_state.uploaded_csv:
                st.warning("Please upload a CSV first before processing.")
            else:
                with st.spinner("Processing"):
                    dfs = []
                    for csv in st.session_state.uploaded_csv:
                        dfs.append(pd.read_csv(csv))

                    # Concatenate all data into one DataFrame
                    concat_frame = pd.concat(dfs, ignore_index=True)
                    st.session_state.df = concat_frame
                    csv_tool = Custom_Tools.get_csv_retrieval_chain(st.session_state.df)

                    
                    

def main():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("faiss_index", embeddings)

    #init  local vector storage 
    if not vectorstore:
        vectorstore = FAISS.from_texts(texts=[""], embedding=embeddings) 
        vectorstore.save_local("faiss_index")

    st.set_page_config(page_title="Craft LLM Automation", page_icon=":stars:")
    #remove the streamlit logo
    st.write(css, unsafe_allow_html=True)
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = None
    if "uploaded_csv" not in st.session_state:
        st.session_state.uploaded_csv = None
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()

    st.header("Craft LLM POC:")
    user_question = st.text_input("Ask a question about your documents:")

    handle_document_upload()
    handle_csv_upload()

    if user_question:
        handle_userinput(user_question)

    
if __name__ == '__main__':
    main()
