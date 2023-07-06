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

from dotenv import load_dotenv

working_directory = TemporaryDirectory()
toolkit = FileManagementToolkit(
    root_dir=str(working_directory.name)
)

from custom_tools import Custom_Tools
local_documents = Custom_Tools.get_document_retrieval_chain()

load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
toolkit.get_tools()
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

#Agent Tools
#tools = load_tools(["llm-math", "serpapi"], llm=llm)

tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
    Tool(
        name="Local_Documents",
        func=local_documents.run,
        description="useful for when you need to answer questions about local documents, documents stored in vector storage, or uploaded documents"
    )
    # Tool(
    #     name="ReadFileTool",
    #     func=ReadFileTool.run,
    #     description="useful for reading folders and files on the local file system"
    # ),
    # Tool(
    #     name="DeleteFileTool",
    #     func=DeleteFileTool.run,
    #     description="useful for deleting folders and files on the local file system"
    # ),
    # Tool(
    #     name="WriteFileTool",
    #     func=WriteFileTool.run,
    #     description="useful for writing folders and files on the local file system"
    # )
]