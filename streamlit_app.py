__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import base64
import uuid

import streamlit as st
import pandas as pd

from pdf2jpg import pdf2jpg
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

def create_tmp_sub_folder():
    """
    Creates a temporary sub folder under tmp

    :return:
    """
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    tmp_sub_folder_name = str(uuid.uuid4())[:8]
    tmp_sub_folder_path = os.path.join("tmp", tmp_sub_folder_name)
    os.mkdir(tmp_sub_folder_path)
    return tmp_sub_folder_path

def write_pdf(pdf_path, pages):
    # Create temporary folder for generated image
    tmp_sub_folder_path = create_tmp_sub_folder()

    # Save images in that sub-folder
    result = pdf2jpg.convert_pdf2jpg(pdf_path, tmp_sub_folder_path, pages=pages)
    images = []
    # for image_path in result[0]["output_jpgfiles"]:
    #     images.append(np.array(Image.open(image_path)))

    # # Create merged image from all images + remove irrelevant whitespace
    # merged_arr = np.concatenate(images)
    # merged_arr = crop_white_space(merged_arr)
    # merged_path = os.path.join(tmp_sub_folder_path, "merged.jpeg")
    # Image.fromarray(merged_arr).save(merged_path)

    # Display the image
    st.image(result[0]["output_jpgfiles"][0]) #(merged_path)


def main():

    def form_upload_callback():
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

                if splits:
                    # create the vectorestore to use as the index
                    st.session_state['vectorstore'] = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
                    # expose this index in a retriever interface
                    retriever = st.session_state['vectorstore'].as_retriever()
                    # create a chain to answer questions 
                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    st.session_state['rag_chain'] = create_retrieval_chain(retriever, question_answer_chain)

        st.write('Finished uploading...')

    
    with st.form(key='uploader'):
        uploaded_docs = st.file_uploader('Upload trial documents in PDF format.',
                                 type=["pdf",],
                                 accept_multiple_files=True)

        uploader_button = st.form_submit_button(label='Process files', type="primary", on_click=form_upload_callback)


    # check if chain is ready before letting user ask questions    
    if 'rag_chain' in st.session_state:
        with st.form(key="questions"):
            question = st.write(
                "Now ask a question about the documents!")
            question = st.text_input('Question:')
            retrieve = st.form_submit_button("Ask", type="primary")
            if retrieve:
                with st.spinner('Thinking...'):
                    st.session_state['results'] = st.session_state['rag_chain'].invoke({"input": question})
                    st.write(st.session_state['results']['answer'])
        
        with st.form(key="sources"):    
            st.write("Sources:")
            st.session_state['source_selector'] = 1

            def render_source_callback():
                idx = int(st.session_state['source_selector']) - 1
                st.write('File name: ' + st.session_state['results']['context'][idx].metadata['source'])
                page_num = int(st.session_state['results']['context'][idx].metadata['page']) + 1
                st.write('Page number: %d' % page_num)
                # pdf_viewer(st.session_state['results']['context'][idx].metadata['source'],
                #            width=900, 
                #            height=1400, 
                #            pages_to_render=[st.session_state['results']['context'][idx].metadata['page']+1],
                #            key='pdf'+str(idx)
                # )
                write_pdf(st.session_state['results']['context'][idx].metadata['source'], page_num)

            st.session_state['source_selector'] = st.selectbox(
                "Select most relevant document fragments to view.",
                ("1", "2", "3", "4"),
                index=0,
                placeholder="Select a source...",
            )

            render_source = st.form_submit_button(label='Render source', on_click=render_source_callback)

                    # if source_selector:
                    #     idx = int(source_selector) - 1
                    #     st.write('File name: ' + results['context'][idx].metadata['source'])
                    #     page_num = int(results['context'][idx].metadata['page']) + 1
                    #     st.write('Page number: %d' % page_num)
                    #     pdf_viewer(results['context'][idx].metadata['source'],
                    #                width=900, 
                    #                height=1400, 
                    #                pages_to_render=[results['context'][idx].metadata['page']+1],
                    #                key='pdf'+str(idx)
                    #     )         
                        # if results:
                        #     # write results to session state
                        #     # call the fragment to render them
                        #     st.session_state['results'] = results
                        #     render_results()
                        # else:
                        #     st.write("No results found: try a different question or upload different documents!")

if __name__ == "__main__":
    main()