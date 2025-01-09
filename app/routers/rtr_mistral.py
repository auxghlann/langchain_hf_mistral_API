from fastapi import APIRouter
from pydantic import BaseModel
from app.core.mistral import Mistral

mistral_router = APIRouter(prefix="/chat")

class Prompt(BaseModel):
    input: str

@mistral_router.post("/prompt")
def generate_response():
    ...


