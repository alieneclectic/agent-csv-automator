from langchain.tools import BaseTool
from typing import Optional
from utils import DocumentStorage
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class Local_Documents(BaseTool):
    name = "Local_Documents"
    description = "useful for when you need to ask a question directly to documents that are uploaded locally in the system"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        return DocumentStorage.get_document_retrieval_chain({"question": query})

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        raise NotImplementedError("Local_Documents does not support async")