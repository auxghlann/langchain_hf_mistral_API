from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.mistral import Mistral


mistral_router: APIRouter = APIRouter(prefix="/chat")

mistral_instance: Mistral = Mistral()

class Prompt(BaseModel):
    query: str
    pdf_path: str

@mistral_router.post("/prompt")
def generate_response(prompt: Prompt) -> dict:

    try:
        query: str = prompt.query
        pdf_path: str = prompt.pdf_path
        response: str = mistral_instance.generate_response(query=query, pdf_path=pdf_path)
        if response:
            return {
                "message": response
            }
        else:
            raise HTTPException(status_code=500, detail="Could not generate response due to error")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


