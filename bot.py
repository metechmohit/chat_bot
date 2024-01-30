import streamlit as st
import pickle
#from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
import os

# Sidebar contents
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space  # Make sure to import this

with st.sidebar:
    st.title('LegalEase Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot where clients upload their documents and query with their documents.
    ''')
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:")

    add_vertical_space(5)  # Ensure that add_vertical_space is properly imported or defined
    st.write('Made with love by LegalEase')

#load_dotenv()

# Use the provided API key
os.environ["OPENAI_API_KEY"] = api_key

def main():
    st.header("Chat with Legal Doc 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your Doc", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()
