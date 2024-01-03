import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
import sys


import os
import docx
import PyPDF2

def read_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

import PyPDF2

def read_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        # Replace PdfFileReader with PdfReader
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text


def read_text_files_from_folder(folder_path):
    all_text = ''

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if file_name.endswith('.txt'):
            all_text += read_text_from_txt(file_path)
        elif file_name.endswith('.docx'):
            all_text += read_text_from_docx(file_path)
        elif file_name.endswith('.pdf'):
            all_text += read_text_from_pdf(file_path)

    return all_text

# Replace 'your_folder_path' with the path to your folder containing .txt, .docx, and .pdf files
folder_path = './files/'


def load_docs(directory):
  loader = DirectoryLoader(directory)
  data = loader.load()
  return data


#Get multiple pdf's as input:
def get_pdf_text(pdf_docs):
    data = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            data += page.extract_text()
    return data

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data in chunks
#def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def chunk_data(data):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store
    #vector_store = Chroma.from_documents(chunks, embeddings)

    # if you want to use a specific directory for chromadb
    # vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
    return vector_store

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    
    vector_store = Chroma.from_documents(chunks, embeddings)

    # if you want to use a specific directory for chromadb
    # vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
    #vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer


# calculate embedding cost using tiktoken
#def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.image('multi_docs.png')
    st.subheader('Chat with your data in multiple formats(Pdf,.txt,.docx) Application')
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        else:
            st.warning("No valid OpenAI API key provided. Please provide it to proceed.")
            sys.exit()    


        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)


        data = read_text_files_from_folder(folder_path)
        chunks = chunk_data(data)
        st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                #tokens, embedding_cost = calculate_embedding_cost(chunks)
                #st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
        vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
        st.session_state.vs = vector_store
        st.success('File uploaded, chunked and embedded successfully.')

    # user's question text input widget
    q = st.text_input('Ask a question about the content of your file:')
    if q: # if the user entered a question and hit enter
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k)

            # text area widget for the LLM answer
            st.text_area('LLM Answer: ', value=answer)

            st.divider()

            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''

            # the current question and answer
            value = f'Q: {q} \nA: {answer}'

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)

