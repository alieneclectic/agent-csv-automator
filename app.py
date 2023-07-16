import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template
from utils import DocumentStorage, DocumentProcessing
from custom_tools import Custom_Tools
from agent import Agent
from pathlib import Path
import numpy as np
import pandas as pd
import openai
import os


def handle_userinput(user_question):
    if user_question:
        #print(st.session_state.conversation)
        #if(st.session_state.conversation == None):
        #agent = Agent.initialize_llama_index_agent()
        agent = Agent.initialize_conversational_agent()
        st.session_state.conversation = agent

        # Truncate or summarize the conversation history if it's too long
        if len(st.session_state.chat_history) > 4000:
            st.session_state.chat_history = st.session_state.chat_history[-4000:]
            
        try:
            #call the OpenAI API
            response = st.session_state.conversation({'input': user_question})
            # Append the response to the existing chat history
            if 'chat_history' in response:
                #st.session_state.chat_history.extend(response['chat_history'])
                st.session_state.chat_history = response['chat_history']
        except openai.error.InvalidRequestError as e:
            # Handle the error
            st.error(f"Error: {str(e)}")

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


def handle_user_qa_input(user_qa_question):
    
    response = st.session_state.qa_conversation({'input': user_qa_question})
    st.write(response)


def list_all_docs():
    dir = "docs"
    return os.listdir(dir)


def write_bullets(data):
    bullet_list = "\n".join(f"- {item}" for item in data)
    st.write(bullet_list)


def delete_all_docs(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))


def send_docs_to_qa(uploadedfile):
     delete_all_docs("qa")
     for doc in uploadedfile:
        with open(os.path.join("qa",doc.name),"wb") as f:
         f.write(doc.getbuffer())


def send_docs_to_index(uploadedfile):
     delete_all_docs("docs")
     for doc in uploadedfile:
        with open(os.path.join("docs",doc.name),"wb") as f:
         f.write(doc.getbuffer())
     st.session_state.llama_index = DocumentStorage.set_llama_index()


def handle_document_upload():
    with st.sidebar:
        st.subheader("Documents")
        # st.write("Uploaded documents:")
        # filelist = list_all_docs()
        # write_bullets(filelist)
        uploaded_docs = st.file_uploader(
            "Upload documents here", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        if uploaded_docs:
            st.session_state.uploaded_docs = uploaded_docs
            with st.spinner("Processing"):
                index = send_docs_to_index(uploaded_docs)
                

                    
                          
def handle_csv_upload():
    with st.sidebar:
        st.subheader("CSVs")
        uploaded_csv = st.file_uploader(
            "Upload CSVs here", type=['csv'], accept_multiple_files=True)
        if uploaded_csv:
            st.session_state.uploaded_csv = uploaded_csv
            with st.spinner("Processing"):
                dfs = []
                for csv in st.session_state.uploaded_csv:
                    dfs.append(pd.read_csv(csv))

                # Concatenate all data into one DataFrame
                concat_frame = pd.concat(dfs, ignore_index=True)
                st.session_state.df = concat_frame
                csv_tool = Custom_Tools.get_pandas_dataframe_agent(st.session_state.df)


def page_tabs():
    tab1, tab2, tab3 = st.tabs(["Data Chat", "Data QA", "Data Visualization",])

    with tab1:
        st.header("Data Chat")
        user_question = st.text_input("Upload and ask a question about your documents:")

        index_type = st.checkbox("Semantic Search", value=st.session_state.index_type)

        if index_type != st.session_state.index_type:
            st.session_state.index_type = index_type
            if index_type:
                st.session_state.query_type = "vector_index"
            else:
                st.session_state.query_type = "list_index"
            

        if user_question:
            handle_userinput(user_question)

    with tab2:
        st.header("Data QA")
        uploaded_qa_docs = st.file_uploader(
            "Upload documents here", type=['csv'], accept_multiple_files=True)
        if uploaded_qa_docs:
            st.session_state.uploaded_qa_docs = uploaded_qa_docs
            with st.spinner("Processing"):
                dfs = []
                for csv in st.session_state.uploaded_qa_docs:
                    dfs.append(pd.read_csv(csv))

                # Concatenate all data into one DataFrame
                concat_frame = pd.concat(dfs, ignore_index=True)
                st.session_state.df_qa = concat_frame
                csv_qa_tool = Custom_Tools.get_pandas_dataframe_agent(st.session_state.df_qa)
                st.session_state.qa_conversation = csv_qa_tool

        user_qa_question = st.text_input("Data QA questions")
        if user_qa_question:
            handle_user_qa_input(user_qa_question)

    with tab3:
        st.header("Data Visualization")
        # chart_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        
        # st.area_chart(chart_data)

    
def main():
    embeddings = OpenAIEmbeddings()
    
    if Path('faiss_index').exists():
        vectorstore = FAISS.load_local("faiss_index", embeddings)
    else:
        vectorstore = None

    #init  local vector storage 
    if not vectorstore:
        vectorstore = FAISS.from_texts(texts=[""], embedding=embeddings) 
        vectorstore.save_local("faiss_index")

    st.set_page_config(page_title="Craft LLM Automation")
    #remove the streamlit logo
    st.write(css, unsafe_allow_html=True)
    streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}

            .block-container::before {
                content: url(https://www.craftww.com/wp-content/themes/CRAFT/assets/images/craft-logo.svg);
                position: absolute;
                top: -80px;
                left: 0px;
                width: 200px;
                height: 50px;
                filter: invert(1) grayscale(1) brightness(2);
            }
            </style>
            """
    st.markdown(streamlit_style, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "qa_conversation" not in st.session_state:
        st.session_state.qa_conversation = None
    if 'index_type' not in st.session_state:
        st.session_state.index_type = False
    if "query_type" not in st.session_state:
        st.session_state.query_type = "list_index"
    if "vector_index" not in st.session_state:
        st.session_state.vector_index = None
    if "list_index" not in st.session_state:
        st.session_state.list_index = None    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = None
    if "uploaded_qa_docs" not in st.session_state:
        st.session_state.uploaded_qa_docs = None
    if "uploaded_csv" not in st.session_state:
        st.session_state.uploaded_csv = None
    if "llama_index" not in st.session_state:
        st.session_state.llama_index = None
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if "df_qa" not in st.session_state:
        st.session_state.df_qa = pd.DataFrame()

    handle_document_upload()
    handle_csv_upload()
    page_tabs()

if __name__ == '__main__':
    main()
