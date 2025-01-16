from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.mistral import Mistral


mistral_router: APIRouter = APIRouter(prefix="/chat")

mistral_instance: Mistral = Mistral()

class Prompt(BaseModel):
    input: str

@mistral_router.post("/prompt")
def generate_response(prompt: Prompt) -> dict:

    try:
        input: str = prompt.input
        response: str = mistral_instance.generate_response(input)
        if response:
            return {
                "message": response
            }
        else:
            raise HTTPException(status_code=500, detail="Could not generate response due to error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


