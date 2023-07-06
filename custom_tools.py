from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

class Custom_Tools():

    def get_document_retrieval_chain():

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local("faiss_index", embeddings)

        document_conversation_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", 
            retriever=vectorstore.as_retriever(),
        )
        return document_conversation_chain
