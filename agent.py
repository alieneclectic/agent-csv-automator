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


llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-16k")

class Agent():
    def initialize_conversational_agent():

        custom_prompt = CustomPromptTemplates.fewShotPromptTemplate1()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        conversational_agent = initialize_agent(
            tools=Agent_Tools.initTools(),
            llm=llm,
            agent="chat-conversational-react-description",
            #agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            #agent=AgentType.OPENAI_FUNCTIONS,
            #agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=5,
            prompt=custom_prompt,
            #system_message = "Add agent message",
            #retriever=vectorstore.as_retriever(),
            handle_parsing_errors=True,
            #early_stopping_method='generate',
            #agent_instructions="Try the 'CSV_Data' or 'Local_Documents' tool first, Use the other tools if relevent and neccessary. Use the Calculator for any math problems",
            memory=memory,
        )
        return conversational_agent
    

    def initialize_llama_index_agent():
        custom_prompt = CustomPromptTemplates.fewShotPromptTemplate1()
        memory = ConversationBufferMemory(memory_key="chat_history")
        llama_index_agent = initialize_agent(
            tools=Agent_Tools.initTools(),
            llm=llm,
            handle_parsing_errors=True,
            #PromptTemplate=custom_prompt,
            agent="conversational-react-description",
            memory=memory
        )
        return llama_index_agent

    
    def initialize_dataframe_agent(df):

        dataframe_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )

        return dataframe_agent
    
    def initialize_sql_agent(db):

        #db = SQLDatabase.from_uri("notebooks/Chinook.db")
        toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

        sql_agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
        )
        return sql_agent


    def initialize_csv_agent(file):

        csv_agent = create_csv_agent(
            OpenAI(temperature=0),
            file,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

        return csv_agent
