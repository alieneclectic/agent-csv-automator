import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.agents import AgentType, create_pandas_dataframe_agent, create_sql_agent, create_csv_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.memory import ConversationBufferMemory
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
from langchain import PromptTemplate, LLMChain
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


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
llm2 = OpenAI(temperature=0, model="gpt-3.5-turbo")

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
    
    
    def generate_csv_data(query):

        #filepath = Path('output/' + file_name)

        document_chain = Custom_Tools.get_document_retrieval_qa_chain()
        csv_chain = Custom_Tools.get_pandas_dataframe_agent(st.session_state.df)

        document_data = document_chain.run('Using all the content from the uploaded documents, return all the text in a human redable format. Only return the test and nothing else.')
        csv_data = st.session_state.df.to_csv()

        template = """
        Use the following CSV data: {csv_data} and the following text data to generate new, CSV data: {document_data} The data generated should be informed by the text data and must exactly match the structure of the original CSV data. Follow any additional guidance from the following query: {query}
        """
        prompt = PromptTemplate(template=template, input_variables=["csv_data", "document_data", "query"])

        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt
        )

        response = llm_chain.predict(csv_data=csv_data, document_data=document_data, query=query)

        return response
    


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