import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template
from utils import DocumentStorage, DocumentProcessing
from custom_tools import Custom_Tools
from agent import Agent
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import fastapi
import openai
import os

app = fastapi()

class API:
    
    @app.post("/")
    def getSomthing():
        return True