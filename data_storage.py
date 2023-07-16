import logging
import sys
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from pathlib import Path
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv

from llama_index import GPTVectorStoreIndex, download_loader

import openai
import os
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


documents = SimpleDirectoryReader(Path("docs")).load_data()
index = VectorStoreIndex.from_documents(documents=documents)


# GoogleSheetsReader = download_loader('GoogleSheetsReader')
# loader = GoogleSheetsReader()
# documents = loader.load_data()
# index = GPTVectorStoreIndex.from_documents(documents)
# index.query('When am I meeting Gordon?')


tools = [
    Tool(
        name="LlamaIndex",
        func=lambda q: str(index.as_query_engine().query(q)),
        description="useful for when you want to answer questions about local documents. The input to this tool should be a complete english sentence.",
        return_direct=True,
    ),
]
# set Logging to DEBUG for more detailed outputs
memory = ConversationBufferMemory(memory_key="chat_history")
llm = ChatOpenAI(temperature=0)
agent_executor = initialize_agent(
    tools, llm, agent="conversational-react-description", memory=memory
)

# agent_executor.run("generate verizon product headlines for each state using the local documents as guidance. The headline should include something about the state or the state name itslef. The output format should be in CSV. The column name for the headline should be 'Headline'")

agent_executor.run("summarize the uploaded documents")