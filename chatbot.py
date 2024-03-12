# Import required modules
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

from langchain_community.chat_models import ChatOllama

from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import time
loader = PyPDFLoader("example.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=100)
all_splits = text_splitter.split_documents(data)

persist_directory = 'jj'

vectorstore = Chroma.from_documents(
    documents=all_splits, embedding=OllamaEmbeddings(model="mistral"),persist_directory=persist_directory)

vectorstore.persist()

vectorstore = Chroma(persist_directory=persist_directory,
                  embedding_function=OllamaEmbeddings(model="mistral")
                  )

llm = Ollama(base_url="http://localhost:11434",
                                  model="mistral:instruct",
                                  verbose=True,
                                  callback_manager=CallbackManager(
                                      [StreamingStdOutCallbackHandler()])
                                  )

retriever = vectorstore.as_retriever()


template = """
    You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.
    
    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"" 
    """
prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )



qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": prompt,
                "memory": memory,
            }
        )


while True:
    query = input("Ask a question: ")
    response = qa_chain(query)



























