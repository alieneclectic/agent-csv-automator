from agent_tools import tools
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from prompt_templates import CustomPromptTemplates
#import pandas as pd

llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-0613")

class Agent():
    def initialize_conversational_agent():

        custom_prompt = CustomPromptTemplates.fewShotPromptTemplate1()

        conversational_agent = initialize_agent(
            tools=tools,
            llm=llm,
            #agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            #agent=AgentType.OPENAI_FUNCTIONS,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=5,
            prompt=custom_prompt,
            #retriever=vectorstore.as_retriever(),
            handle_parsing_errors=True,
            agent_instructions="Try the 'CSV_Data' or 'Local_Documents' tool first, Use the other tools if relevent and neccessary.",
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