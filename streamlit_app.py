__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
import pandas as pd

from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(layout="wide",
                   page_title="Knowledge Management",
)

"""
# Welcome to the Knowledge Management tool!

Please upload trial docs to begin asking questions. 
"""

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_KEY']
if not os.environ["OPENAI_API_KEY"]:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
llm = ChatOpenAI(model="gpt-4o")
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation"
)   

uploaded_docs = st.file_uploader('Upload trial documents in PDF format.',
                                 type=["pdf",],
                                 accept_multiple_files=True)

system_prompt = (
        """
        You are assisting a clinical research coordinator engaged in a clinical trial for a new drug.
        Use the following pieces of retrieved context from the trial research protocol to answer the question.
        Cite details from the context pertaining to an answer and be very thorough.
        If you don't know the answer, say that you don't know.
        
        "\n\n"
        "{context}"
        """
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

splits = []

if uploaded_docs is not None:
    # parse the docs
    for file in uploaded_docs:
        loader = PyPDFLoader(os.path.join("/tmp", file))
        docs = loader.load()
        # split the documents into chunks
        splits.extend(text_splitter.split_documents(docs))

    # create the vectorestore to use as the index
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    # expose this index in a retriever interface
    retriever = vectorstore.as_retriever()
    # create a chain to answer questions 
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the documents!",
        placeholder="Can you give me a short summary of the protocol?",
        disabled=not uploaded_docs,
    )

    with st.form(key="questions"):
        question = st.text_input('Question to answer:')
        retrieve = st.form_submit_button("Ask", type="primary")
        if retrieve:
            results = rag_chain.invoke({"input": question})
            if results:
                st.write((results['answer']))
                st.write("Sources:")
                for idx, item in enumerate(results['context']):
                    st.write('\nSource %s:' % str(idx+1))
                    st.write(results['context'][idx].metadata)
                    st.write(results['context'][idx].page_content)