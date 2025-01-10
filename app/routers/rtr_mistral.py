from fastapi import APIRouter
from pydantic import BaseModel
from app.core.mistral import Mistral

mistral_router = APIRouter(prefix="/chat")

mistral_instance = Mistral()

class Prompt(BaseModel):
    input: str

@mistral_router.post("/prompt")
def generate_response(prompt: Prompt):

    input: str = prompt.input

    response: str = mistral_instance.generate_response(input)

    if response:
        return {
            "message" : response
        }

    else:
        return {
            "message": "Could not generate response due to error"
        }


