# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from API.utils import initialize_qa_chain, get_full_text_and_ask_question

app = FastAPI(
    title="Langchain Books Project - API",
    description="Une API pour interagir avec l'agent LangChain sur de nombreuses références littéraires",  # noqa
    version="1.0.0"
)


# Modèle de requête pour poser une question
class QueryRequest(BaseModel):
    question: str
    book_title: str = None  # Facultatif
    book_id: int = None  # Facultatif


# Initialiser la chaîne QA lors du démarrage de l'application
qa_chain = initialize_qa_chain()


@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API!"}


@app.post("/query/")
async def query_book(query: QueryRequest):
    try:
        # Poser une question simple
        answer = qa_chain.run(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/book_full_text/")
async def ask_question_on_full_text(query: QueryRequest):
    try:
        if not query.book_title:
            raise HTTPException(status_code=400, detail="Le titre du livre est requis.")  # noqa

        # Poser une question sur le texte complet du livre
        answer = await get_full_text_and_ask_question(query.book_title, query.question)  # noqa
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
