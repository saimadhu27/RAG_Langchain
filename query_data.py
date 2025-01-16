import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import helper.constants as CNT

from RAG_Langchain.get_database import get_embeddings


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text):
    # Prepare the DB.
    embedding_function = get_embeddings()
    db = Chroma(persist_directory=CNT.CHROMA_PATH, embedding_function=embedding_function)
        
    query_embedding = embedding_function.embed_query(query_text)

    #print(f"Query embedding: {query_embedding}")
    
    # Search the DB and return the top 5 answers
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    print(results)
    # all_docs = db.get(include=["documents"])
    # for doc in all_docs["documents"]:
    #     print(doc[:200])  # Print a snippet of each document

    # existing_items = db.get(include=[])  # IDs are always included by default
    # existing_ids = set(existing_items["ids"])
    # print(f"Number of existing documents in DB: {len(existing_ids)}")
    # print(existing_ids)
    if not results:
        print("No results found.")
        return "No relevant documents found."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(f"Context text:\n{context_text}")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    #print(results)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()