import pandas as pd
import os
from dotenv import load_dotenv
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import GutenbergLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pickle

load_dotenv()

gutenberg_data = pd.read_csv("Data/gutenberg_ebooks.csv")

azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY_4")
azure_openai_api_endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT_4")
azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME_4")

embedding_model = AzureOpenAIEmbeddings(
    openai_api_key=azure_openai_api_key,
    deployment="text-embedding-3-large",
    azure_endpoint=azure_openai_api_endpoint,
    openai_api_version="2023-05-15",
    chunk_size=500
)

prompt_template = """
Vous êtes un assistant expert en littérature.

Utilisez les informations suivantes sur les livres pour répondre à la question de manière précise et détaillée.

Informations :
{context}

Question :
{question}

Réponse :
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# Fonction pour charger l'index FAISS
def load_index(index_path):
    """Charger l'index FAISS avec le modèle d'embeddings."""
    vectorstore = FAISS.load_local(
        index_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True  # Sinon erreur
    )
    return vectorstore


# Fonction pour initialiser la chaîne QA
def initialize_qa_chain():
    """Initialisation du retriever et de la chaîne de Questions / Réponses."""
    vectorstore = load_index("book_index")
    retriever = vectorstore.as_retriever()

    llm = AzureChatOpenAI(
        api_key=azure_openai_api_key,
        api_version="2023-12-01-preview",
        azure_endpoint=azure_openai_api_endpoint,
        model=azure_deployment_name,
        temperature=0
    )

    # Création de la chaîne QA avec le prompt personnalisé
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain


# Fonction pour générer l'URL du texte du livre à partir de Gutenberg
def generate_gutenberg_url(book_id):
    """
    Génèrer l'URL pour télécharger le texte d'un livre à partir de Project Gutenberg.  # noqa
    
    Args:
        book_id (int): ID du livre dans le projet Gutenberg.
    
    Returns:
        str: L'URL du fichier texte.
    """
    base_url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    return base_url


async def get_full_text_and_ask_question(title, question):
    """
    Récupérer le texte complet d'un livre et pose une question sur ce même texte.  # noqa

    Args:
        title (str): Titre du livre.
        question (str): Question à poser.

    Returns:
        dict: Réponse de l'agent et un extrait du texte.
    """
    # Récupération des livres correspondants
    matching_books = gutenberg_data[gutenberg_data["Title"].str.lower() == title.lower()]  # noqa

    if matching_books.empty:
        return {"answer": f"Aucun livre intitulé '{title}' n'a été trouvé."}

    book_id = matching_books.iloc[0]["EBook-No."]
    vectorstore_path = f"vectorstores/vectorstore_{book_id}"
    documents_path = f"vectorstores/documents_{book_id}.pkl"  # Fichier pour sauvegarder les documents  # noqa

    split_documents = None 

    # On vérifie si le vectorstore existe déjà
    if os.path.exists(vectorstore_path):
        full_txt_vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        print(f"Vectorstore chargé depuis {vectorstore_path}.")

        # Charger les documents à partir du fichier pickle si disponible
        if os.path.exists(documents_path):
            with open(documents_path, "rb") as f:
                split_documents = pickle.load(f)
                print(f"Documents chargés depuis {documents_path}.")
        else:
            # Si le fichier pickle n'existe pas, on le recrée à partir du vectorstore  # noqa
            split_documents = full_txt_vectorstore.index_to_documents.values()
            with open(documents_path, "wb") as f:
                pickle.dump(split_documents, f)
                print(f"Documents sauvegardés dans {documents_path}.")
    else:
        # S'il n'existe pas, on va charger le texte complet depuis Project Gutenberg  # noqa
        url = generate_gutenberg_url(book_id)
        try:
            loader = GutenbergLoader(url)
            full_txt_documents = loader.load()
            print(f"Texte complet du livre {book_id} chargé avec succès depuis {url}")  # noqa
        except Exception as e:
            # Essayer une URL alternative en cas d'erreur
            alternative_url = f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt"  # noqa
            try:
                loader = GutenbergLoader(alternative_url)
                full_txt_documents = loader.load()
                print(f"Texte complet du livre {book_id} chargé avec succès depuis {alternative_url}")  # noqa
            except Exception as e:
                return {"answer": f"Impossible de charger le texte complet : {e}"}  # noqa

        # Division du document en parties avec text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100
        )
        split_documents = text_splitter.split_documents(full_txt_documents)

        try:
            full_txt_vectorstore = FAISS.from_documents(split_documents, embedding_model)  # noqa
            full_txt_vectorstore.save_local(vectorstore_path)
            print(f"Vectorstore du livre sauvegardé dans {vectorstore_path}.")

            # Sauvegarder les documents dans un fichier pickle
            with open(documents_path, 'wb') as f:
                pickle.dump(split_documents, f)
        except Exception as e:
            return {"answer": f"Erreur lors de la création du vectorstore: {e}"}

    # Initialisation du retriever et de l'agent de questions/réponses
    retriever = full_txt_vectorstore.as_retriever()

    llm = AzureChatOpenAI(
        api_key=azure_openai_api_key,
        api_version="2023-12-01-preview",
        azure_endpoint=azure_openai_api_endpoint,
        model=azure_deployment_name,
        temperature=0
    )

    qa_chain_full_text = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

    try:
        # Réponse à la question
        answer = qa_chain_full_text.run(question)

        # Retourner l'extrait si les documents sont chargés, sinon indiquer qu'aucun extrait n'est disponible  # noqa
        if split_documents:
            # Créer un extrait du texte
            full_text = "\n".join([doc.page_content for doc in split_documents])  # noqa
            return {
                "answer": answer,
                "text_extract": full_text[:1000]  # Retourner un extrait
            }
        else:
            # Si les docs ne sont pas disponibles, retourner tout de même la réponse  # noqa
            return {
                "answer": answer,
                "text_extract": "L'extrait de texte est indisponible"  # noqa
            }
    except Exception as e:
        return {"answer": f"Impossible de générer une réponse : {e}"}
