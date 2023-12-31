# Agent CSV Generator

## Introduction
------------
Agent CSV Generator is a Python application that allows you to chat with multiple documents. These documents can be used to tward the generation oc CSV data feeds. 

## How It Works
------------

The application follows these steps to provide responses to your questions:

1. Document Loading: The app reads multiple documents and extracts their text content.

2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

3. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

4. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the documents.

## Dependencies and Installation
----------------------------
To install the Agent CSV Generator App, please follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

3. Obtain an API key from OpenAI and add it to the `.env` file in the project directory.

## Usage
-----
To use the Agent CSV Generator App, follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file.

2. Run the `main.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Load multiple documents into the app by following the provided instructions.

5. Ask questions in natural language about the loaded documents using the chat interface.

## License
-------
The Agent CSV Generator App is released under the [MIT License](https://opensource.org/licenses/MIT).