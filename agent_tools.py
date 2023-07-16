import streamlit as st
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import load_tools
from langchain.agents.agent_toolkits import FileManagementToolkit, PlayWrightBrowserToolkit
from tempfile import TemporaryDirectory
from custom_tools import Custom_Tools
from dotenv import load_dotenv
import pandas as pd
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser,
)

# async_browser = create_sync_playwright_browser()
# browser_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser).get_tools()
# click_element, navigate_browser, previous_webpage, extract_text, get_elements, current_webpage = browser_toolkit

working_directory = TemporaryDirectory()
file_toolkit = FileManagementToolkit(
    root_dir=str('/Users/jason.english/Documents/GitHub/agent-csv-automator/output'),
    selected_tools=["read_file", "write_file", "list_directory"]
).get_tools()
read_tool, write_tool, list_tool = file_toolkit
load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)


if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()


#Agent Tools
class Agent_Tools:
    def initTools():
        tools = [
            # read_tool,
            # write_tool,
            # list_tool,
            # click_element,
            # navigate_browser,
            # previous_webpage,
            # extract_text,
            # get_elements,
            # current_webpage,
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="useful for when you need to answer questions about math. Input should be a valid numerical expression"
            ),
            # Tool(
            #     name="Local_Documents_Chain",
            #     func=Custom_Tools.get_document_retrieval_qa_chain().run,
            #     description="useful for when you need to answer questions about local documents, documents stored in vector storage, or uploaded documents. Input should be a fully formed question.",
            #     return_direct=True
            # ),
            # Tool(
            #     name="LlamaIndex",
            #     func=lambda q: str(index.as_query_engine().query(q)),
            #     description="useful for when you want to answer questions about local documents. The input to this tool should be a complete english sentence.",
            #     return_direct=True,
            # ),
            Tool(
                name="CSV_Data_Retrival_Tool",
                func=Custom_Tools.get_pandas_dataframe_agent(st.session_state.df).run,
                description="useful for when you need to answer questions about CSV documents that were uploaded or manipulate a CSVs data stored in a dataframe. This tool leverages the Pandas framwork. Input should be a fully formed question.",
                return_direct=True
            ),
            Tool(
                name = "CSV_Generator_Tool",
                func=Custom_Tools.generate_csv_data,
                description="useful for when you need create or generate CSV data, especially based data from the Local_Documents_Chain and the CSV_Data_Chain",
                return_direct=True
            ),
            Tool(
                name="Send_To_Google_Sheets_Tool",
                func=Custom_Tools.send_to_google_sheets,
                description="useful for when you need to send or update CSV data in Google Sheets."
            ),
            Tool(
                name="Llama_Index_Agent",
                func=lambda q: str(st.session_state[st.session_state.query_type].as_query_engine().query(q)),
                description="useful for when you need to answer questions about local documents, documents stored in vector storage, or uploaded documents, summarizing document information. Input should be a fully formed question.",
                return_direct=True,
            ),
            # Tool(
            #     name="Zapier_Agent",
            #     func=Custom_Tools.get_zapier_agent().run,
            #     description="useful for when you need to interact with the Zapier API and NLA API requests.",
            #     return_direct=True
            # ),
            # Tool(
            #     name="SQL_Data_Agent",
            #     func=Custom_Tools.get_sql_agent,
            #     description="useful for when you need to answer questions about SQL data, or external SQL data using an robust LLM agent. Input should be a fully formed question."
            # ),
            Tool(
                name = "Search",
                func=search.run,
                description="useful for when you need to answer questions about current events. You should ask targeted questions"
            )
        ]

        return tools
