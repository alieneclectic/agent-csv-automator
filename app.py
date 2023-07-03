import streamlit as st
from htmlTemplates import css, bot_template, user_template
from utils import DocumentStorage, DocumentProcessing
from agent import Agent

def handle_userinput(user_question):
    response = st.session_state.conversation({'input': user_question})
    st.session_state.chat_history = response['chat_history']

    # Reverse the chat history in pairs
    reversed_chat_history = list(reversed([st.session_state.chat_history[i:i+2] for i in range(0, len(st.session_state.chat_history), 2)]))
    flattened_chat_history = [item for sublist in reversed_chat_history for item in sublist]

    print(flattened_chat_history)
    
    for i, message in enumerate(flattened_chat_history):
        if i % 2 == 0:
            print(message.content)
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            print(message.content)
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Craft LLM Automation", page_icon=":books:")
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
    if "document_history" not in st.session_state:
        st.session_state.document_history = []
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = None

    st.header("Craft LLM Automation POC:")
    user_question = st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_docs = st.file_uploader(
            "Upload your documents here and click on 'Process'", type=['pdf', 'docx', 'txt', 'csv'], accept_multiple_files=True)
        if uploaded_docs:
            st.session_state.uploaded_docs = uploaded_docs
        if st.button("Upload & Process"):
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

                    text_chunks = DocumentProcessing.get_text_chunks(raw_text, 500, 30)

                    # create vector store
                    vectorstore = DocumentStorage.get_vectorstore(text_chunks, doc.type, doc.name)

                    # create conversation chain
                    st.session_state.conversation = Agent.get_conversation_chain(vectorstore)

    if user_question:
        if st.session_state.conversation is None:
            st.warning("Please upload and process a document first before asking a question.")
        else:
            handle_userinput(user_question)

    # # Add a "Clear Chat" button
    # if st.button("Clear Chat"):
    #     st.session_state.chat_history = None


if __name__ == '__main__':
    main()