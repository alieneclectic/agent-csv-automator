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
from datetime import datetime
import pandas as pd
import io
from pathlib import Path
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

import gspread
from dotenv import load_dotenv
from langchain import SerpAPIWrapper

search = SerpAPIWrapper()
load_dotenv()


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
#llm2 = OpenAI(temperature=0, model="gpt-3.5-turbo")

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
            llm=llm,
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

        document_chain = Custom_Tools.get_document_retrieval_qa_chain()
        #csv_chain = Custom_Tools.get_pandas_dataframe_agent(st.session_state.df)

        # prepare local document text
        document_data = document_chain.run('Using all the content from the uploaded documents. Only return the text and nothing else.')
        csv_data = st.session_state.df.to_csv(index=False)

        template = """
        Use the following CSV data: {csv_data} and text data examples as a reference: {document_data} for context. Generate new CSV data informed by the text data example. The CSV data should exactly match the columns names of the CSV data example. Follow any additional guidance from the following query: {query}. The output must be in CSV format. 
        """
        prompt = PromptTemplate(template=template, input_variables=["csv_data", "document_data", "query"])

        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose = True
        )

        # create local file
        filepath = Path('output/feed.csv')
        response = llm_chain.run(csv_data=csv_data, document_data=document_data, query=query)
        new_df = pd.read_csv(io.StringIO(response), sep=",")
        new_df.to_csv(filepath, index=False)


        #send to Google Sheets with gspread
        now = datetime.now()
        ct = now.strftime("%m/%d/%Y, %H:%M:%S")
        filepath = Path('service_account.json')
        gc = gspread.oauth(credentials_filename=filepath)
        sh = gc.open("DCO AI Generated")
        worksheet = sh.add_worksheet(title="DCO Feed-" + ct, rows=52, cols=10)
        sheet_df = pd.read_csv(Path('output/feed.csv'))
        worksheet.update([sheet_df.columns.values.tolist()] + sheet_df.values.tolist())
        #print(sh.sheet1.get('A1'))

        


        # send to Google Sheets with Zapier
        # zapier = Custom_Tools.get_zapier_agent()
        # zapier_template = """
        # create a new google sheets using the following csv data {new_df}
        # """
        # zapier.run(zapier_template)

        #Example prompts
        # Summarize the uploaded [doc_name] document
        # Generate a new row of csv data for each state on the west coast of the USA. The Headline column should include include the state name
        # Generate a new row of csv data for each state in the mid west of the USA. The Headline column should include include the state name
        # Generate a new row of csv data for each state in the East Coast of the USA. The Headline column should include include the state name
        # Generate a new row of csv data for each county in the the state of New Jersey and make the Headline column include the county name

        return response