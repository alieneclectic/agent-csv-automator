from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.agents import create_pandas_dataframe_agent
from langchain.vectorstores import FAISS
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")


class Custom_Tools():

    def get_document_retrieval_chain():

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local("faiss_index", embeddings)

        document_conversation_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        return document_conversation_chain

    def get_csv_retrieval_chain(df):

        csv_retrieval_chain = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

        return csv_retrieval_chain