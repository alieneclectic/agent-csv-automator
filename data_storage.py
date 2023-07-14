import logging
import sys
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

from llama_index import VectorStoreIndex, SimpleDirectoryReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

documents = SimpleDirectoryReader("/Users/jason.english/Desktop/verizon-poc/pdfs").load_data()
print(documents)
index = VectorStoreIndex.from_documents(documents=documents)

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