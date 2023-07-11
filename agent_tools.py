import streamlit as st
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import load_tools
from langchain.agents.agent_toolkits import FileManagementToolkit
from tempfile import TemporaryDirectory
from custom_tools import Custom_Tools
from dotenv import load_dotenv
import pandas as pd

working_directory = TemporaryDirectory()
file_toolkit = FileManagementToolkit(
    root_dir=str('/Users/jason.english/Documents/GitHub/agent-csv-automator/output'),
    selected_tools=["read_file", "write_file", "list_directory"]
).get_tools()
read_tool, write_tool, list_tool = file_toolkit
load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)


if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()


#Agent Tools
class Agent_Tools:
    def initTools():
        tools = [
            read_tool,
            write_tool,
            list_tool,
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="useful for when you need to answer questions about math. Input should be a valid numerical expression"
            ),
            Tool(
                name="Local_Documents_Chain",
                func=Custom_Tools.get_document_retrieval_qa_chain().run,
                description="useful for when you need to answer questions about local documents, documents stored in vector storage, or uploaded documents. Input should be a fully formed question.",
                return_direct=True
            ),
            Tool(
                name="CSV_Data_Agent",
                func=Custom_Tools.get_pandas_dataframe_agent(st.session_state.df).run,
                description="useful for when you need to answer questions about CSV documents that were uploaded or manipulate a CSVs data stored in a dataframe. This tool leverages the Pandas framwork. Input should be a fully formed question.",
                return_direct=True
            ),
            Tool(
                name="Zapier_Agent",
                func=Custom_Tools.get_zapier_agent().run,
                description="useful for when you need to interact with the Zapier API and NLA API requests.",
                return_direct=True
            ),
            Tool(
                name = "CSV_Generator",
                func=Custom_Tools.generate_csv_data,
                description="useful for when you need create a CSV file based on gathered information from the Local_Documents_Chain and the CSV_Data_Chain",
                return_direct=True
            ),
            # Tool(
            #     name="SQL_Data_Agent",
            #     func=Custom_Tools.get_sql_agent,
            #     description="useful for when you need to answer questions about SQL data, or external SQL data using an robust LLM agent. Input should be a fully formed question."
            # ),
            # Tool(
            #     name="CSV_Data_Agent",
            #     func=Custom_Tools.get_csv_agent_retrieval_chain,
            #     description="useful for when you need to answer questions about CSV data, or an uploaded CSV using an robust LLM agent. Input should be a fully formed question."
            # ),
            # Tool(
            #     name = "Search",
            #     func=search.run,
            #     description="useful for when you need to answer questions about current events. You should ask targeted questions"
            # )
        ]

        return tools
