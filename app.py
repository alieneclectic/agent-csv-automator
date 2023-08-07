import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template
from utils import DocumentStorage, DocumentProcessing
from custom_tools import Custom_Tools
from agent import Agent
from pathlib import Path
from llama_index.evaluation import ResponseEvaluator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import openai
import os
import gspread


def handle_userinput(user_question):
    if user_question:

        # Check if there are any documents uploaded
        if not st.session_state.uploaded_docs:
            st.error("Please upload documents before asking a question.")
            return      

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
        st.subheader("Upload to Vector Database")
        uploaded_docs = st.file_uploader(
            "Upload documents here", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        if uploaded_docs and not st.session_state.get('docs_uploaded', False):
            st.session_state.uploaded_docs = uploaded_docs
            with st.spinner("Processing"):
                index = send_docs_to_index(uploaded_docs)
            st.session_state.docs_uploaded = True

def handle_csv_upload():
    with st.sidebar:
        st.subheader("CSVs Uploader")
        uploaded_csv = st.file_uploader(
            "Upload CSVs here", type=['csv'], accept_multiple_files=True)
        if uploaded_csv and not st.session_state.get('csv_uploaded', False):
            st.session_state.uploaded_csv = uploaded_csv
            with st.spinner("Processing"):
                dfs = []
                for csv in st.session_state.uploaded_csv:
                    dfs.append(pd.read_csv(csv))

                # Concatenate all data into one DataFrame
                concat_frame = pd.concat(dfs, ignore_index=True)
                st.session_state.df = concat_frame
                csv_tool = Custom_Tools.get_pandas_dataframe_agent(st.session_state.df)
            st.session_state.csv_uploaded = True

google_creds_filepath = Path('service_account.json')
def get_all_rows_from_sheet(sheet_name):
    # Authenticate with Google Sheets using the service account
    
    gc = gspread.oauth(credentials_filename=google_creds_filepath)
    
    # Open the desired spreadsheet
    sh = gc.open("DCO AI Generated")
    
    # Select the desired worksheet
    worksheet = sh.worksheet(sheet_name)
    
    # Fetch all rows from the worksheet
    all_rows = worksheet.get_all_values()

    # Convert the list of lists into a pandas DataFrame
    # The first row is considered as the header
    df = pd.DataFrame(all_rows[1:], columns=all_rows[0])
    
    return df


def sync_to_google_sheets(df, sheet_name):
    # Authenticate with Google Sheets using the service account
    gc = gspread.oauth(credentials_filename=google_creds_filepath)
    
    # Open the desired spreadsheet
    sh = gc.open("DCO AI Generated")
    
    # Select the desired worksheet
    worksheet = sh.worksheet(sheet_name)
    
    # Clear the existing content
    worksheet.clear()
    
    # Update the worksheet with the new data
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())


def page_tabs():
    tab1, tab2, tab3, tab4 = st.tabs(["Data Chat", "Feed Manager", "Data QA", "Data Visualization"])

    with tab1:
        st.header("Data Chat")
        user_question = st.text_input("Upload and ask a question about your documents:")

        index_type = st.checkbox("Full Search", value=st.session_state.index_type)

        if index_type != st.session_state.index_type:
            st.session_state.index_type = index_type
            if index_type:
                st.session_state.query_type = "list_index"
            else:
                st.session_state.query_type = "vector_index"
            

        if user_question:
            handle_userinput(user_question)

    with tab2:
        st.header("Feed Manager")
        df = get_all_rows_from_sheet('DCO Feed-07/24/2023, 07:44:23')
        modified_df = st.data_editor(df)

        # Sync the modified data to Google Sheets
        if st.button("Sync Data"):
            sync_to_google_sheets(modified_df, 'DCO Feed-07/24/2023, 07:44:23')
            st.success("Data synced to Google Sheets successfully!")

    with tab3:
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

    with tab4:
        st.header("Data Visualization")
        user_visualization_question = st.text_input("Data visualization questions")

        if user_visualization_question:
            question = user_qa_question
            #handle_user_visualization_input(user_qa_question)

        st.write("Data visualization")
        chart_data = pd.DataFrame(
            np.random.randn(200, 3),
            columns=['a', 'b', 'c'])

        st.vega_lite_chart(chart_data, {
            'mark': {'type': 'circle', 'tooltip': True},
            'encoding': {
                    'x': {'field': 'a', 'type': 'quantitative'},
                    'y': {'field': 'b', 'type': 'quantitative'},
                    'size': {'field': 'c', 'type': 'quantitative'},
                    'color': {'field': 'c', 'type': 'quantitative'},
                }
            },
            use_container_width = True
        )

        # arr = np.random.normal(1, 1, size=100)
        # fig, ax = plt.subplots()
        # ax.hist(arr, bins=20)

        # st.pyplot(fig)

    
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
                /* filter: invert(1) grayscale(1) brightness(2); */
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
        st.session_state.query_type = "vector_index"
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

    agent = Agent.initialize_conversational_agent()
    st.session_state.conversation = agent

    handle_document_upload()
    handle_csv_upload()
    page_tabs()



if __name__ == '__main__':
    main()
