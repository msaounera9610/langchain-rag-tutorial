from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings


CHROMA_PATH = "chroma"

#query_text = 'i want to check if my colum is not null and that is an iban'
#query_text = 'je veux verifier que ma colonne est un numéro de téléphone'
query_text = 'validate phone number'
#model_id = 'sentence-transformers/all-MiniLM-L6-v2'
model_id = 'mixedbread-ai/mxbai-embed-large-v1'
embedding_function = HuggingFaceInferenceAPIEmbeddings(api_key='',
                                                api_url=f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
                                                )
query_text = f'Represent this sentence for searching relevant passages: {query_text}'
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Search the DB.
results = db.similarity_search_with_relevance_scores(query_text, k=3)
#print(results)
for i, result in enumerate(results):
    print(f"result {i}:\n{result}\n")
