from agent_tools import tools
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

class Agent():
    def get_conversation_chain(vectorstore):
        conversation_chain = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            #agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            max_iterations=len(tools),
            #agent_instructions="Try 'Knowledge Internal Base' tool first, Use the other tools if these don't work.",
            #early_stopping_method='generate',
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        )
        return conversation_chain