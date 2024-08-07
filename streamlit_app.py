__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import base64
import uuid

import streamlit as st
import pandas as pd

from streamlit_pdf_viewer import pdf_viewer
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
# Welcome to the clinical trial knowledge management tool!

Please upload trial docs to begin asking questions. 
"""

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_KEY']
if not os.environ["OPENAI_API_KEY"]:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
llm = ChatOpenAI(model="gpt-4o")
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation"
)   

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


if 'pages' not in st.session_state:
    st.session_state['pages'] = None

if 'page_selection' not in st.session_state:
    st.session_state['page_selection'] = []


with st.sidebar:
    st.header("Page Selection")
    placeholder = st.empty()
    
    if not st.session_state['pages']:
        st.session_state['page_selection'] = placeholder.multiselect(
            "Select pages to display",
            options=[],
            default=[],
            help="The page number considered is the PDF number and not the document page number.",
            disabled=not st.session_state['pages'],
            key=1
        )


def upload_callback():
    with st.spinner('Processing...'):
        # clean up the state
        # eg del the vector store and qachain if already intialized with prior doc set
        for key in st.session_state.keys():
            del st.session_state[key]

        if uploaded_docs is not None:
            # save uploaded files to disk
            for i in range(len(uploaded_docs)):
                bytes_data = uploaded_docs[i].read()  # read the content of the file in binary
                # print(uploaded_docs[i].name, bytes_data)
                with open(os.path.join("/tmp", uploaded_docs[i].name), "wb") as f:
                    f.write(bytes_data)  # write this content elsewhere

            # parse the files and add to the splits collection
            splits = []
            for file in uploaded_docs:
                with st.spinner('Loading...'):
                    loader = PyPDFLoader(os.path.join("/tmp", file.name))
                    docs = loader.load()
                    # split the documents into chunks
                    splits.extend(text_splitter.split_documents(docs))
                    #TODO will eventually need to support multidocs
                    st.session_state['pages'] = len(docs) #if not st.session_state['pages'] else st.session_state['pages']

            if splits:
                # create the vectorestore to use as the index
                st.session_state['vectorstore'] = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
                # expose this index in a retriever interface
                retriever = st.session_state['vectorstore'].as_retriever()
                # create a chain to answer questions 
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                st.session_state['rag_chain'] = create_retrieval_chain(retriever, question_answer_chain)

    st.write('Finished uploading...')

def invoke_chain_callback():
    with st.spinner('Thinking...'):
        st.session_state['results'] = st.session_state['rag_chain'].invoke({"input": question})
        st.write('Chain returned an answer...')

            
with st.form(key='uploader'):
    uploaded_docs = st.file_uploader('Upload trial documents in PDF format.',
                             type=["pdf",],
                             accept_multiple_files=True)

    uploader_button = st.form_submit_button(label='Process files', type="primary", on_click=upload_callback)

# check if chain is ready before letting user ask questions 
asked = None   
if 'rag_chain' in st.session_state:
    with st.form(key="questions"):
        question = st.write("Now ask a question about the documents!")
        question = st.text_input('Question:')
        asked = st.form_submit_button("Ask", type="primary", on_click=invoke_chain_callback)                
                
if asked:
    st.write(st.session_state['results']['answer'])
    st.write("Sources:")
    source_pages = []
    source_list = ['\nSource %s:' % str(idx+1) for idx,_ in enumerate(st.session_state['results']['context'])]                            
    for idx, item in enumerate(st.session_state['results']['context']):
        st.write(source_list[idx])
        st.write('File name: ' + st.session_state['results']['context'][idx].metadata['source'])
        page_num = int(st.session_state['results']['context'][idx].metadata['page']) + 1
        st.write('Page number: %d' % page_num)
        source_pages.append(page_num)

    if st.session_state['pages']:
        st.session_state['page_selection'] = placeholder.multiselect(
            "Select pages to display",
            options=list(range(1, st.session_state['pages'] + 1)),
            default=source_pages,
            help="The page number considered is the PDF number and not the document page number.",
            disabled=not st.session_state['pages'],
            key=2
        )

    pdf_viewer(st.session_state['results']['context'][idx].metadata['source'],
               width=900, 
               height=1400, 
               pages_to_render=st.session_state['page_selection'],
               key='pdf_'+str(idx))