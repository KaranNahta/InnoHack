import os
import sys
import logging
import chromadb
from chromadb.utils import embedding_functions
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("vector_store")

COLLECTION_NAME = "regulatory_precedents"

# Legal precedents to seed
PRECEDENTS = [
    {
        "id": "precedent_01",
        "text": "Essential Commodities Act (Section 3) - Section 3 of the Essential Commodities Act, 1955 empowers the Central Government to control the production, supply, distribution, and pricing of essential commodities. Wholesalers and retailers who artificially spike prices, hoard stock, or create artificial shortages are subject to search, seizure, license cancellation, and prosecution."
    },
    {
        "id": "precedent_02",
        "text": "Regulatory Intervention Policy and Ceiling Warnings - Under statutory price control guidelines, if the daily modal price of an essential commodity (such as Potato, Tomato, or Onion) breaches the calibrated statutory ceiling (p90), the regulator is authorized to issue warning notices to registered vendors demanding explanation of cost increases and ordering compliance."
    },
    {
        "id": "precedent_03",
        "text": "Antitrust Cartel and Price Fixing Regulations - Section 3 of the Competition Act, 2002 prohibits horizontal price-fixing cartels. If three or more independent wholesalers in the same region exhibit synchronized price increases above 2.5 standard deviations without corresponding spikes in transport or wholesale indices, it is treated as cartel behavior, resulting in antitrust audits."
    },
    {
        "id": "precedent_04",
        "text": "Statutory Price Controls Amendment 2023 - Strict statutory ceilings are enforced on staple vegetables during seasonal production shortages. Interventions are triggered automatically when regional retail or wholesale modal price averages exceed historic seasonal indexes by more than 20%."
    }
]

def populate_database(db_path: str = "data/chroma") -> None:
    """
    Initializes a persistent ChromaDB client, embeds the regulatory precedents,
    and inserts them into the collection.
    """
    logger.info("Initializing persistent ChromaDB client at %s...", db_path)
    os.makedirs(db_path, exist_ok=True)
    
    client = chromadb.PersistentClient(path=db_path)
    
    # Embedding function
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )
    
    # Prepare documents
    ids = [p["id"] for p in PRECEDENTS]
    documents = [p["text"] for p in PRECEDENTS]
    metadatas = [{"source": "Essential Commodities ECA guidelines"} for _ in PRECEDENTS]
    
    # Add to collection
    logger.info("Adding %d legal precedents to ChromaDB collection '%s'...", len(documents), COLLECTION_NAME)
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    logger.info("ChromaDB vector store seeded successfully.")

def query_precedents(query_text: str, n_results: int = 2, db_path: str = "data/chroma") -> List[str]:
    """
    Queries ChromaDB for legal precedents relevant to the query text.
    Automatically initializes the database if it doesn't exist.
    """
    if not os.path.exists(os.path.join(db_path, "chroma.sqlite3")) and not os.path.exists(db_path):
        logger.info("Chroma DB not found. Seeding first...")
        populate_database(db_path)
        
    client = chromadb.PersistentClient(path=db_path)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )
    
    logger.info("Querying vector store for: '%s'...", query_text)
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    # Extract documents from results
    documents = results.get("documents", [[]])[0]
    return list(documents)

def main():
    # CLI entry point to seed database and test queries
    db_path = "data/chroma"
    populate_database(db_path)
    
    # Test query
    test_query = "synchronized pricing behavior of three wholesalers"
    docs = query_precedents(test_query, n_results=2, db_path=db_path)
    logger.info("Query results:")
    for idx, d in enumerate(docs):
        logger.info("[%d] %s", idx+1, d)

if __name__ == "__main__":
    main()
