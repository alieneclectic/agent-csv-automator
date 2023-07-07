from langchain.prompts import PromptTemplate
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

class CustomPromptTemplates:
    def fewShotPromptTemplate1():
        template = "You are a creative strategist in the advertising agency space. You also have a background in data science. You are currently in an environment where CSV, PDF, DOC, and TXT files will be uploaded and available locally. The PDF, DOC, and TXT data can be accessed through agent tool called Local_Documents. The CSV Data can be access through a tool called CSV_Data."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        example_human = HumanMessagePromptTemplate.from_template("How many columns are in the csv data?")
        example_ai = AIMessagePromptTemplate.from_template("There are 10 columns in the csv data")
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, example_human, example_ai, human_message_prompt]
        )

        return chat_prompt