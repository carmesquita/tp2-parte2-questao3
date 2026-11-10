from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel


#Modelo dos dados recebidos pela API
class TextRequest(BaseModel):
    prompt: str

#função para carregar o modelo de tradução
def load_translation_model():
    pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de", device="cpu")
    model = HuggingFacePipeline(pipeline=pipe)
    return model

#tradução do texto
def translate_text(model, text):
    translation = model.invoke(text)
    return translation