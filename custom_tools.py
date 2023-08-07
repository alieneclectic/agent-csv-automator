import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.agents import AgentType, Tool, AgentExecutor, create_pandas_dataframe_agent, create_sql_agent, create_csv_agent, initialize_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit, ZapierToolkit
from langchain.memory import ConversationBufferMemory
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate, LLMChain
from datetime import datetime
import pandas as pd
import io
from pathlib import Path
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import gspread
from dotenv import load_dotenv
from langchain import SerpAPIWrapper

search = SerpAPIWrapper()
load_dotenv()


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

class Custom_Tools():

    def get_document_retrieval_qa_chain():

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local("faiss_index", embeddings)

        document_conversation_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            verbose=True,
            retriever=vectorstore.as_retriever()
        )

        return document_conversation_chain

    def get_pandas_dataframe_agent(df):

        csv_retrieval_chain = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

        return csv_retrieval_chain
    

    def get_llama_index_agent():
        index = st.session_state.llama_index
        tools = [
            Tool(
                    name="LlamaIndex",
                    func=lambda q: str(index.as_query_engine().query(q)),
                    verbose=True,
                    description="useful for when you want to answer questions about local documents. The input to this tool should be a complete english sentence.",
                    return_direct=True,
                ),
        ]
        # set Logging to DEBUG for more detailed outputs
        memory = ConversationBufferMemory(memory_key="chat_history")
        llm = ChatOpenAI(temperature=0)
        llama_agent = initialize_agent(
            tools, llm, agent="conversational-react-description", memory=memory,
            verbose=True
        )

        return llama_agent


    def get_sql_agent(db):

        toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

        sql_retrieval_chain = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
        )
        return sql_retrieval_chain
    

    def get_csv_agent(file):

        csv_retrieval_chain = create_csv_agent(
            OpenAI(temperature=0),
            file,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )

        return csv_retrieval_chain
    

    def get_zapier_agent():

        zapier = ZapierNLAWrapper()
        toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
        zapier_agent = initialize_agent(
            toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )

        return zapier_agent
    

    def get_sql_index_tool(sql_index, table_context_dict):
        table_context_str = "\n".join(table_context_dict.values())

        def run_sql_index_query(query_text):
            try:
                response = sql_index.as_query_engine(synthesize_response=False).query(query_text)
            except Exception as e:
                return f"Error running SQL {e}.\nNot able to retrieve answer."
            text = str(response)
            sql = response.extra_info["sql_query"]
            return f"Here are the details on the SQL table: {table_context_str}\nSQL Query Used: {sql}\nSQL Result: {text}\n"
            # return f"SQL Query Used: {sql}\nSQL Result: {text}\n"

        return run_sql_index_query 
    
    
    def generate_csv_data(query):

        document_chain = Custom_Tools.get_document_retrieval_qa_chain()
        #csv_chain = Custom_Tools.get_pandas_dataframe_agent(st.session_state.df)

        # prepare local document text
        document_data = document_chain.run('Using all the content from the uploaded documents. Only return the text and nothing else.')
        csv_data = st.session_state.df.to_csv(index=False)

        filepath = Path('output/feed.csv')
        
        template = """
        Use the following CSV data: {csv_data} and text data examples as a reference: {document_data} for context. Generate completely new CSV data informed by the text data example. The CSV data should exactly match the columns names of the CSV data example. Follow any additional guidance from the following query: {query}. Do not output the CSV directly, only confirm the task is complete and create a link to the file using this path 
        """
        prompt = PromptTemplate(template=template, input_variables=["csv_data", "document_data", "query"])

        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose = True
        )

        # create local file
        
        response = llm_chain.run(csv_data=csv_data, document_data=document_data, query=query)
        new_df = pd.read_csv(io.StringIO(response), sep=",")
        new_df.to_csv(filepath, index=False)

        return response
    

    def send_to_google_sheets(query):

        #create a new tab and send to Google Sheets
        now = datetime.now()
        title = title="DCO Feed-"
        ct = now.strftime("%m/%d/%Y, %H:%M:%S")
        filepath = Path('service_account.json')
        gc = gspread.oauth(credentials_filename=filepath)
        sh = gc.open("DCO AI Generated")
        worksheet = sh.add_worksheet(title=title + ct, rows=100, cols=12)
        sheet_df = pd.read_csv(Path('output/feed.csv'))
        sheet_df.fillna("", inplace=True)
        if not sheet_df.empty:
            try:
                worksheet.update([sheet_df.columns.values.tolist()] + sheet_df.values.tolist())
                return "A new sheet called" + title + ct + "has been generated and sent to Google Sheets successfuly."
            except Exception as e:
                return f"Error running send to Google Sheets {e}."
        else:
            return "No csv data found."

    
    def get_sql_index_tool(sql_index, table_context_dict):
        table_context_str = "\n".join(table_context_dict.values())

        def run_sql_index_query(query_text):
            try:
                response = sql_index.as_query_engine(synthesize_response=False).query(query_text)
            except Exception as e:
                return f"Error running SQL {e}.\nNot able to retrieve answer."
            text = str(response)
            sql = response.extra_info["sql_query"]
            return f"Here are the details on the SQL table: {table_context_str}\nSQL Query Used: {sql}\nSQL Result: {text}\n"
            # return f"SQL Query Used: {sql}\nSQL Result: {text}\n"

        return run_sql_index_query
    

#Example prompts
# Summarize the uploaded [doc_name] document

# Generate a new row of csv data for each state on the West coast of the USA. The Headline column should include include the state name

# Generate a new row of csv data for each state in Middle America of the USA. The Headline column should include include the state name

# Generate a new row of csv data for each state in the East Coast of the USA. The Headline column should include include the state name

# Generate a new row of csv data for each county in the the state of New Jersey and make the Headline column include the county name

# Generate a new row of csv data for each in the the state in the United States and make the Headline column include the state name