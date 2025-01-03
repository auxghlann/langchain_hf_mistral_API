import time
from typing import (
    Iterator,
    List
)
from langchain_huggingface import HuggingFaceEndpoint
from requests.exceptions import HTTPError
# from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
import os
from dotenv import load_dotenv

load_dotenv()

class Mistral:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Mistral, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
            self.api_token = os.environ.get("HF_API_KEY")
            if not self.api_token:
                raise ValueError("HF_API_KEY environment variable not set")
            # print(f"Using API Token: {self.api_token}")
            self.initialized = True


    def __initialize_hf_llm(self) -> HuggingFaceEndpoint:
        llm = HuggingFaceEndpoint(
            # endpoint_url=self.endpoint_url,
            repo_id = self.repo_id,
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.03,
            task="text-generation",
            huggingfacehub_api_token= self.api_token,
        )

        return llm
    
    #TODOLIST:
    # [load document (pdf)]
    def __get_pdf_document(self, pdf_path: str) -> Iterator[Document]:
        pdf_doc: PyPDFLoader = PyPDFLoader(file_path=pdf_path)
        return pdf_doc.lazy_load()

    # [split text]

    def __split_text_from_doc(self, load_doc: Iterator[Document]) -> List[Document]:
        text_splitter: RecursiveCharacterTextSplitter = \
             RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(load_doc)
    
    # [embbed]


    # [vector store]
    # chain document

    #TODO: import create_stuff_document_chain
    # create chain
    # create retriever and retrieval chain
    # call llm
    
    def __llm_parse_output(self, behavior: str, input: str) -> str:
        llm = self._initialize_hf_llm()

        prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                ("system", "{behavior}"),
                ("human", "{input}")
            ]
        )

        parser: StrOutputParser = StrOutputParser()
        chain = prompt | llm | parser
        
        return chain.invoke({
            "behavior": behavior,
            "input" : input
        })



    def generate_response(self, behavior:str, prompt: str) -> str:
        pdf_path: str = \
            "C:\\Users\\Khester Mesa\\Documents\\projects\\langchain_hf_mistral_API\\app\\src\\pdf_resume\\Résumé_Mesa (1).pdf"
        doc = self.__get_pdf_document(pdf_path=pdf_path)
        split_text = self.__split_text_from_doc(doc)

        print(len(split_text))

        # return self._llm_parse_output(behavior, prompt)

    def test_response(self):
        pdf_path: str = \
            "C:\\Users\\Khester Mesa\\Documents\\projects\\langchain_hf_mistral_API\\app\\src\\pdf_resume\\Résumé_Mesa (1).pdf"
        doc = self.__get_pdf_document(pdf_path=pdf_path)
        split_text = self.__split_text_from_doc(doc)

        print(len(split_text))

        for i in split_text:
            print(i.page_content)

