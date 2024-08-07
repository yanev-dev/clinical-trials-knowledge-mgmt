__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import base64

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

def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def main():

    def form_upload_callback():
        with st.spinner('Processing...'):
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

                if splits:
                    if 'vectorstore' not in st.session_state:
                        # create the vectorestore to use as the index
                        st.session_state['vectorstore'] = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
                        # expose this index in a retriever interface
                        retriever = st.session_state['vectorstore'].as_retriever()
                        # create a chain to answer questions 
                        question_answer_chain = create_stuff_documents_chain(llm, prompt)
                        st.session_state['rag_chain'] = create_retrieval_chain(retriever, question_answer_chain)

    with st.form(key='uploader'):
        uploaded_docs = st.file_uploader('Upload trial documents in PDF format.',
                                 type=["pdf",],
                                 accept_multiple_files=True)

        uploader_button = st.form_submit_button(label='Process Files', on_click=form_upload_callback)

    with st.form(key="questions"):
        question = st.write(
            "Now ask a question about the documents!")
        question = st.text_input('Question:')
        retrieve = st.form_submit_button("Ask", type="primary")
        if retrieve and uploader_button:
            with st.spinner('Thinking...'):
                results = st.session_state['rag_chain'].invoke({"input": question})
                if results:
                    st.write((results['answer']))
                    st.write("Sources:")
                    tabs_list = st.tabs(['\nSource %s:' % str(idx+1) for idx,_ in enumerate(results['context'])])                            
                    for idx, item in enumerate(results['context']):
                        with tabs_list[idx]:
                            st.write('File name: ' + results['context'][idx].metadata['source'])
                            page_num = int(results['context'][idx].metadata['page']) + 1
                            st.write('Page number: %d' % page_num)
                            pdf_viewer(results['context'][idx].metadata['source'],
                                       width=900, 
                                       height=1400, 
                                       pages_to_render=[results['context'][idx].metadata['page']+1],
                                       key='pdf'+str(idx))
        else:
            st.write("No results found: try a different question or upload different documents!")

if __name__ == "__main__":
    main()