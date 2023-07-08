import streamlit as st
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import load_tools
from langchain.tools.file_management import (
    ReadFileTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    WriteFileTool,
    ListDirectoryTool
)
from langchain.agents.agent_toolkits import FileManagementToolkit
from tempfile import TemporaryDirectory
from custom_tools import Custom_Tools
from agent import Agent
from dotenv import load_dotenv
import pandas as pd

working_directory = TemporaryDirectory()
toolkit = FileManagementToolkit(
    root_dir=str(working_directory.name)
)

load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
toolkit.get_tools()
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
read_file = ReadFileTool()
write_file = WriteFileTool()
delete_file = DeleteFileTool()
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()


#Agent Tools

class Agent_Tools:
    def initTools():
        tools = [
            # Tool(
            #     name = "Search",
            #     func=search.run,
            #     description="useful for when you need to answer questions about current events. You should ask targeted questions"
            # ),
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="useful for when you need to answer questions about math. Input should be a valid numerical expression"
            ),
            Tool(
                name="Local_Documents_Chain",
                func=Custom_Tools.get_document_retrieval_chain().run,
                description="useful for when you need to answer questions about local documents, documents stored in vector storage, or uploaded documents. Input should be a fully formed question."
            ),
            Tool(
                name="CSV_Data_Chain",
                func=Custom_Tools.get_csv_retrieval_chain(st.session_state.df).run,
                description="useful for when you need to answer questions about csv documents, or an uploaded csv. Input should be a fully formed question."
            ),
            Tool(
                name="SQL_Data_Agent",
                func=Agent.initialize_sql_agent().run,
                description="useful for when you need to answer questions about sql data, or an uploaded csv. Input should be a fully formed question."
            ),
            Tool(
                name="CSV_Data_Agent",
                func=Agent.initialize_csv_agent().run,
                description="useful for when you need to answer questions about csv data, or an uploaded csv. Input should be a fully formed question."
            ),
            Tool(
                name="ReadFileTool",
                func=read_file.run,
                description="useful for reading folders and files on the local file system."
            ),
            Tool(
                name="DeleteFileTool",
                func=delete_file.run,
                description="useful for deleting folders and files on the local file system."
            )
        ]

        return tools
