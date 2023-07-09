import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.agents import AgentType, create_pandas_dataframe_agent, create_sql_agent, create_csv_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from typing import Optional, Type
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
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
    

    def get_sql_agent_retrieval_chain(db):

        toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

        sql_retrieval_chain = create_sql_agent(
            llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
        )
        return sql_retrieval_chain
    

    def get_csv_agent_retrieval_chain(db):

        csv_retrieval_chain = create_csv_agent(
            OpenAI(temperature=0),
            "titanic.csv",
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

        return csv_retrieval_chain
    
