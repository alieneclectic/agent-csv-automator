from agent_tools import tools
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
#import pandas as pd

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

class Agent():
    def initialize_conversational_agent():

        conversational_agent = initialize_agent(
            tools=tools,
            llm=llm,
            #agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            #agent=AgentType.OPENAI_FUNCTIONS,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=5,
            #retriever=vectorstore.as_retriever(),
            handle_parsing_errors=True,
            #agent_instructions="Try 'Knowledge Internal Base' tool first, Use the other tools if these don't work.",
            #early_stopping_method='generate',
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        )

        return conversational_agent
    
    def initialize_dataframe_agent(df):
        
        #df = pd.read_csv("titanic.csv")

        dataframe_agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )

        return dataframe_agent