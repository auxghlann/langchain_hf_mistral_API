import time
from langchain_community.llms import HuggingFaceEndpoint
from requests.exceptions import HTTPError
import os
from dotenv import load_dotenv

load_dotenv()

class Mistral:
    def __init__(self):
        self.endpoint_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/"
        self.api_token = os.environ.get("HF_API_KEY")
        if not self.api_token:
            raise ValueError("HF_API_KEY environment variable not set")
        print(f"Using API Token: {self.api_token}")

    def generate_response(self, prompt: str) -> str:
        llm = HuggingFaceEndpoint(
            endpoint_url=self.endpoint_url,
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.03,
            task="text-generation",
            huggingfacehub_api_token=self.api_token,
        )

        for _ in range(5):  # Retry up to 5 times
            try:
                response = llm(prompt)
                return response
            except HTTPError as e:
                if e.response.status_code == 429:
                    print("Rate limit exceeded. Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    raise e
        raise Exception("Failed to get a response after 5 retries")