from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
import os
import shutil
import re


CHROMA_PATH = "chroma"
#DATA_PATH = "data/books"


def extract_docstrings(code: str) -> list:
    docstring_pattern = re.compile(r'""".*?"""', re.DOTALL)
    return docstring_pattern.findall(code)

def extract_function_names(code: str) -> list:
    function_pattern = re.compile(r'def\s+(?P<func_name>\w+)\s*\(.*?\)\s*->\s*.*?:', re.DOTALL)
    
    matches = function_pattern.finditer(code)
    function_names = [match.group('func_name') for match in matches]

    return function_names


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    with open('functions_calista.txt', 'r+') as file:
        doc = file.read()
        return doc


def split_text(document):
    chunks = []
    docstrings = extract_docstrings(document)
    funcs = extract_function_names(document)
    for i, docstring in enumerate(docstrings):
        metadata = {"start_index": funcs[i]}
        chunks.append(Document(page_content=docstring, metadata=metadata))

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    #model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    model_id = 'mixedbread-ai/mxbai-embed-large-v1'
    model = HuggingFaceInferenceAPIEmbeddings(api_key='',
                                                  api_url=f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
                                                  )

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, model, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    generate_data_store()
