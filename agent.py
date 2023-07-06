from agent_tools import tools
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

class Agent():
    def initialize_agent():

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local("faiss_index", embeddings)

        conversation_chain = initialize_agent(
            tools=tools,
            llm=llm,
            #agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            #agent=AgentType.OPENAI_FUNCTIONS,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=5,
            retriever=vectorstore.as_retriever(),
            #agent_instructions="Try 'Knowledge Internal Base' tool first, Use the other tools if these don't work.",
            #early_stopping_method='generate',
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        )

        return conversation_chain