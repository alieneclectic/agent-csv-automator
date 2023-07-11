import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.agents import AgentType, create_pandas_dataframe_agent, create_sql_agent, create_csv_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.agents import initialize_agent
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from typing import Optional, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
from pathlib import Path
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from dotenv import load_dotenv
from langchain import SerpAPIWrapper

search = SerpAPIWrapper()
load_dotenv()


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

class Custom_Tools():

    def get_document_retrieval_qa_chain():

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local("faiss_index", embeddings)

        document_conversation_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            verbose=True,
            retriever=vectorstore.as_retriever()
        )

        return document_conversation_chain

    def get_pandas_dataframe_agent(df):

        csv_retrieval_chain = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

        return csv_retrieval_chain
    

    def get_sql_agent(db):

        toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

        sql_retrieval_chain = create_sql_agent(
            llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
        )
        return sql_retrieval_chain
    

    def get_csv_agent(db):

        csv_retrieval_chain = create_csv_agent(
            OpenAI(temperature=0),
            "test.csv",
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )

        return csv_retrieval_chain
    

    def get_zapier_agent():

        zapier = ZapierNLAWrapper()
        toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
        zapier_agent = initialize_agent(
            toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )

        return zapier_agent
    
    
    def generate_csv_file(file_name):
        filepath = Path('output/' + file_name)  
        # df.to_csv(filepath)
        return file_name + ' was created.'
    


# class CustomCSVGeneratorTool(BaseTool):
#     name = "CSV_Generator"
#     description = "useful for when you need create a CSV file based on gathered information from the Local_Documents_Chain and the CSV_Data_Chain."

#     def _run(
#         self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
#     ) -> str:
#         """Use the tool."""
#         response = "CSV Generated"
#         return response

#     async def _arun(
#         self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
#     ) -> str:
#         raise NotImplementedError("custom_search does not support async")