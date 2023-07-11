from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.agents import AgentType, initialize_agent, create_pandas_dataframe_agent, create_sql_agent, create_csv_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from prompt_templates import CustomPromptTemplates
from agent_tools import Agent_Tools
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# template="You are a helpful assistant that translates {input_language} to {output_language}."
# system_message_prompt = SystemMessagePromptTemplate.from_template(template)
# human_template="{text}"
# human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-0613")

class Agent():
    def initialize_conversational_agent():

        custom_prompt = CustomPromptTemplates.fewShotPromptTemplate1()

        conversational_agent = initialize_agent(
            tools=Agent_Tools.initTools(),
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            #agent=AgentType.OPENAI_FUNCTIONS,
            #agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=5,
            prompt=custom_prompt,
            #system_message = "Add agent message",
            #retriever=vectorstore.as_retriever(),
            handle_parsing_errors=True,
            early_stopping_method='generate',
            agent_instructions="Try the 'CSV_Data' or 'Local_Documents' tool first, Use the other tools if relevent and neccessary. Use the Calculator for any math problems",
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True),
        )

        #print(conversational_agent)

        return conversational_agent
    
    def initialize_dataframe_agent(df):

        dataframe_agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )

        return dataframe_agent
    
    def initialize_sql_agent(db):

        #db = SQLDatabase.from_uri("notebooks/Chinook.db")
        toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

        sql_agent = create_sql_agent(
            llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
        )
        return sql_agent


    def initialize_csv_agent():

        csv_agent = create_csv_agent(
            OpenAI(temperature=0),
            "titanic.csv",
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

        return csv_agent
