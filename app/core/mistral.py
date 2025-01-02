import time
from typing import Iterator
from langchain_huggingface import HuggingFaceEndpoint
from requests.exceptions import HTTPError
# from langchain.chains import LLMChain
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


    def _initialize_hf_llm(self) -> HuggingFaceEndpoint:
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

    def _get_pdf_document(self, pdf_path: str) -> Iterator[Document]:
        pdf_doc: PyPDFLoader = PyPDFLoader(file_path="C:\\Users\\Khester Mesa\\Documents\\projects\\langchain_hf_mistral_API\\app\\src\\pdf_resume\\Résumé_Mesa (1).pdf")

        return pdf_doc.lazy_load()

    # [split text]

    def _split_text_from_doc(self, load_doc):
        ...
    # [embbed]
    # [vector store]
    # call llm
    
    def _llm_parse_output(self, behavior: str, input: str) -> str:
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
        
        return self._llm_parse_output(behavior, prompt)

        # return response

        # for _ in range(5):  # Retry up to 5 times
        #     try:
        #         llm_chain = prompt | llm
        #         response = llm(prompt)
        #         return response
        #     except HTTPError as e:
        #         if e.response.status_code == 429:
        #             print("Rate limit exceeded. Retrying in 5 seconds...")
        #             time.sleep(5)
        #         else:
        #             raise e
        # raise Exception("Failed to get a response after 5 retries")