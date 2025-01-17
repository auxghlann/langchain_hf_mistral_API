import time
from typing import (
    Iterator,
    List
)
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables.base import Runnable
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import VectorStore, FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
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
            huggingfacehub_api_token= self.api_token
        )

        return llm

    # [load document (pdf)]
    def __get_pdf_document(self, pdf_path: str) -> Iterator[Document]:
        pdf_doc: PyPDFLoader = PyPDFLoader(file_path=pdf_path)
        return pdf_doc.lazy_load()

    # [split text]
    # TODO: replace recursiveCharacterTextSplitter to -> SemanticSplitter

    def __split_text_from_doc(self, load_doc: Iterator[Document]) -> List[Document]:
        text_splitter: RecursiveCharacterTextSplitter = \
             RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(load_doc)
    
    # [embbed]
    def __initialize_hf_llm_embedding(self) -> HuggingFaceEndpointEmbeddings:
        model = "sentence-transformers/all-mpnet-base-v2"
        embedding: HuggingFaceEndpointEmbeddings = \
            HuggingFaceEndpointEmbeddings(
                model = model,
                task = "feature-extraction",
                huggingfacehub_api_token = self.api_token,
            )

        return embedding

    # [vector store]
    def __vector_store(self, doc: List[Document], embedding: HuggingFaceEndpointEmbeddings) -> VectorStore:
        vector_store: VectorStore = FAISS.from_documents(documents=doc, embedding=embedding)
        return vector_store

    # chain document
    # <deprecated
    # def __llm_parse_output(self, behavior: str, input: str) -> str:
    #     llm = self._initialize_hf_llm()

    #     prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    #         [
    #             ("system", "{behavior}"),
    #             ("human", "{input}")
    #         ]
    #     )

    #     parser: StrOutputParser = StrOutputParser()
    #     chain = prompt | llm | parser
        
    #     return chain.invoke({
    #         "behavior": behavior,
    #         "input" : input
    #     })
    # # >

    # [create retriever and retrieval chain]
    def __create_document_chain(self, llm: HuggingFaceEndpoint):
        parser: StrOutputParser = StrOutputParser()

        prompt = ChatPromptTemplate.from_template("""
                Answer the following question based only on the provided context. 
                Think step by step before providing a detailed answer. 
                I will tip you $1000 if the user finds the answer helpful. 
                <context>
                {context}
                </context>
                Question: {input}""")
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt, output_parser=parser)

        return document_chain

    def __create_retrieval_chain(self, db_vector_store: VectorStore, doc_chain) -> Runnable:

        retriever = db_vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)

        return retrieval_chain

    # [invoke retriever and get "answer" attr]
    def generate_response(self, query: str, pdf_path: str) -> str:

        if query and pdf_path:
            llm: HuggingFaceEndpoint = self.__initialize_hf_llm()
            doc: Iterator[Document] = self.__get_pdf_document(pdf_path=pdf_path)
            split_text: List[Document] = self.__split_text_from_doc(doc)
            embedding: HuggingFaceEndpointEmbeddings = self.__initialize_hf_llm_embedding()
            vector_store: VectorStore = self.__vector_store(split_text, embedding)
            document_chain = self.__create_document_chain(llm=llm)
            retrieval_chain: Runnable = self.__create_retrieval_chain(
                db_vector_store=vector_store,
                doc_chain=document_chain
            )

            response = retrieval_chain.invoke(
                {
                    "input" : query
                }
            )

            return response["answer"]

        raise ValueError("Query and pdf path should not be empty")
        # return self._llm_parse_output(behavior, prompt)


