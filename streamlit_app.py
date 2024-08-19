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
        Use the following pieces of retrieved context from the trial research protocol and supporting documents to answer the question.
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


if 'results' not in st.session_state:
    st.session_state['results'] = None

if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None

if 'rag_chain' not in st.session_state:
    st.session_state['rag_chain'] = None

if 'display_answer' not in st.session_state:
    st.session_state['display_answer'] = None


#### Callbacks ####

def upload_callback():
    with st.spinner('Loading...'):

        #TODO: find a better way to deal with duplicates
        # clean up the state
        # eg del the vector store and qachain if already intialized with prior doc set
        # for key in st.session_state.keys():
        #     st.session_state[key] = None

        if uploaded_docs is not None:
            # save uploaded files to disk
            for i in range(len(uploaded_docs)):
                bytes_data = uploaded_docs[i].read()  # read the content of the file in binary
                # print(uploaded_docs[i].name, bytes_data)
                with open(os.path.join("/tmp", uploaded_docs[i].name), "wb") as f:
                    f.write(bytes_data)  # write this content elsewhere


def invoke_chain_callback():
    with st.spinner('Thinking...'):
        st.session_state['results'] = st.session_state['rag_chain'].invoke({"input": question})


@st.fragment()
def render_pdf_pages():
    if st.session_state['results']:
        st.header('Answer:')
        st.write(st.session_state['results']['answer'])
        st.header("Sources:")

        file_to_pages = {}
        source_list = ['\nSource %s:' % str(idx+1) for idx,_ in enumerate(st.session_state['results']['context'])]                            
        for idx, item in enumerate(st.session_state['results']['context']):
            fname = st.session_state['results']['context'][idx].metadata['source']
            page_num = int(st.session_state['results']['context'][idx].metadata['page']) + 1
            if fname not in file_to_pages:
                file_to_pages[fname] = [page_num]
            else:
                file_to_pages[fname].append(page_num)

        for k,v in file_to_pages.items():
            with st.container():
                st.header("File name: " + k)
                with st.expander("See document source"):
                    st.subheader("Relevant pages:")
                    page_str = ':green[' + ', '.join(str(x) for x in sorted(v)) + ']'
                    cols = st.columns(2)
                    cols[0].markdown(page_str)
                    cols[1].toggle("Refresh", key='toggle_'+k)
                    pdf_viewer(k,
                               width=900, 
                               height=1400, 
                               pages_to_render=v, #st.session_state['page_selection'],
                               key='pdf_'+k)

#### App code ####


with st.form(key='uploader'):
    uploaded_docs = st.file_uploader('Upload trial documents in PDF format.',
                             type=["pdf",],
                             accept_multiple_files=True)

    uploader_button = st.form_submit_button(label='Upload files', type="primary", on_click=upload_callback)
    if uploader_button:
        st.write('Finished uploading...')


if st.button('Process files'):
    # parse the files and add to the splits collection
    splits = []
    with st.spinner('Processing...'):
        for file in uploaded_docs:
            loader = PyPDFLoader(os.path.join("/tmp", file.name))
            docs = loader.load()
            splits.extend(text_splitter.split_documents(docs))
            
        if splits:
            # create the vectorestore to use as the index
            if st.session_state.vector_store is None:
                st.session_state['vector_store'] = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
                # expose this index in a retriever interface
                retriever = st.session_state['vector_store'].as_retriever()
                # create a chain to answer questions 
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
            if st.session_state.rag_chain is None:
                st.session_state['rag_chain'] = create_retrieval_chain(retriever, question_answer_chain)
    st.write('...VectorDB and LLM ready!')


# check if chain is ready before letting user ask questions 
if 'rag_chain' in st.session_state:
    with st.form(key="questions"):
        question = st.write("Now ask a question about the documents!")
        question = st.text_input('Question:')
        asked = st.form_submit_button("Ask", type="primary", on_click=invoke_chain_callback)
        if asked:
            st.write('Chain returned an answer...')
            render_pdf_pages()