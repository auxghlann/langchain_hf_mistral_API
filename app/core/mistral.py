import time
from langchain_huggingface import HuggingFaceEndpoint
from requests.exceptions import HTTPError
# from langchain.chains import LLMChain
# from langchain_core.prompts import PromptTemplate
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


    def _initialize_llm(self, repo_id, api_token) -> any:
        llm = HuggingFaceEndpoint(
            # endpoint_url=self.endpoint_url,
            repo_id = repo_id,
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.03,
            task="text-generation",
            huggingfacehub_api_token=api_token,
        )

        return llm

    def generate_response(self, prompt: str) -> str:
        
        llm = self._initialize_llm(self.repo_id, self.api_token)
    
        response = llm.invoke(prompt)

        return response

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