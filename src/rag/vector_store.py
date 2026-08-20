"""
CASPER-Gov: Persistent Legal Vector Store & Statutory Retriever
===============================================================
Utilizes ChromaDB with sentence-transformers/all-MiniLM-L6-v2 embeddings
to store and retrieve Indian statutory texts, ECA directives, antitrust
regulations, and Legal Metrology rules for court-ready enforcement notices.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("vector_store")

COLLECTION_NAME = "regulatory_precedents"
DEFAULT_DB_PATH = "data/chromadb"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Seed Statutory Precedents & Legal Corpus
# ---------------------------------------------------------------------------
STATUTORY_PRECEDENTS = [
    {
        "id": "eca_1955_sec_3",
        "statute": "Essential Commodities Act, 1955",
        "section": "Section 3",
        "title": "Powers to control production, supply, distribution of essential commodities",
        "text": (
            "Essential Commodities Act 1955 (Section 3: Powers to control production, supply, distribution) - "
            "Section 3 of the Essential Commodities Act, 1955 empowers the Central and State Governments to regulate "
            "or prohibit the production, supply, and distribution of essential commodities, and trade and commerce therein. "
            "Under Section 3(2)(c), orders may be promulgated for controlling the price at which any essential commodity "
            "may be bought or sold. Mandis, wholesalers, and retail vendors engaged in price gouging, artificial price spikes, "
            "or speculative holding are subject to mandatory cost audits, search and seizure of inventory, suspension of trade licenses, "
            "and criminal prosecution under Section 7."
        ),
        "category": "Price Control & Supply Regulation",
    },
    {
        "id": "blackmarketing_act_1980",
        "statute": "Prevention of Blackmarketing and Maintenance of Supplies of Essential Commodities Act, 1980",
        "section": "Section 3 & Section 6",
        "title": "Preventive detention and anti-hoarding enforcement for essential supplies",
        "text": (
            "Prevention of Blackmarketing and Maintenance of Supplies of Essential Commodities Act 1980 - "
            "Authorizes executive authorities and District Magistrates to issue preventive detention orders against any individual "
            "or corporate entity committing, abetting, or instigating acts prejudicial to the maintenance of supplies of essential commodities. "
            "Hoarding, unauthorized diversion of mandi stocks, creating artificial scarcity, or withholding stock from the market to inflate "
            "spot prices constitutes a cognizable offense with attachment of warehouse premises."
        ),
        "category": "Anti-Hoarding & Blackmarketing",
    },
    {
        "id": "legal_metrology_mrp_rules",
        "statute": "Legal Metrology (Packaged Commodities) Rules, 2011 & MRP Enforcement Directives",
        "section": "Rule 18 & Section 36",
        "title": "Maximum Retail Price (MRP) Compliance and Anti-Overcharging Mandates",
        "text": (
            "Legal Metrology (Packaged Commodities) Rules & Maximum Retail Price (MRP) Enforcement Directives - "
            "Strictly prohibits any manufacturer, packer, wholesale distributor, or retail trader from selling or distributing "
            "any essential commodity or packaged staple at a price exceeding the declared Maximum Retail Price (MRP). "
            "Mandatory compliance audits are authorized on point-of-sale registers, and violations incur statutory fines, "
            "confiscation of non-compliant packaged lots, and recurring statutory penalty enhancements for repeated non-compliance."
        ),
        "category": "Retail Price Compliance & Fair Trade",
    },
    {
        "id": "competition_act_2002_sec_3",
        "statute": "Competition Act, 2002",
        "section": "Section 3(3)(a)",
        "title": "Prohibition of Anti-Competitive Agreements & Horizontal Price-Fixing Cartels",
        "text": (
            "Competition Act 2002 (Section 3: Anti-competitive agreements & Cartelization) - "
            "Section 3(3) explicitly declares any agreement or coordinated practice entered into between enterprises, traders, or mandis "
            "engaged in identical or similar trade of goods to be void if it directly or indirectly determines purchase or sale prices. "
            "Synchronized, abnormal price spikes across multiple independent vendors exceeding 2.5 standard deviations without underlying "
            "cost justification trigger an immediate presumption of cartel behavior, attracting Competition Commission audits and severe pecuniary penalties."
        ),
        "category": "Antitrust & Cartelization",
    },
]


def _get_embedding_function():
    """Initializes and returns sentence-transformers embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )


def populate_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initializes a persistent ChromaDB client and seeds statutory texts if empty.
    """
    os.makedirs(db_path, exist_ok=True)
    logger.info("Initializing persistent ChromaDB client at %s...", db_path)
    client = chromadb.PersistentClient(path=db_path)
    embedding_func = _get_embedding_function()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
    )

    # Check if already populated
    if collection.count() == 0:
        logger.info("Seeding %d statutory precedents into ChromaDB collection '%s'...", len(STATUTORY_PRECEDENTS), COLLECTION_NAME)
        ids = [p["id"] for p in STATUTORY_PRECEDENTS]
        documents = [p["text"] for p in STATUTORY_PRECEDENTS]
        metadatas = [
            {
                "statute": p["statute"],
                "section": p["section"],
                "title": p["title"],
                "category": p["category"],
            }
            for p in STATUTORY_PRECEDENTS
        ]
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("ChromaDB vector store seeded successfully (%d documents).", len(ids))
    else:
        logger.info("ChromaDB collection '%s' already contains %d documents. Skipping re-seed.", COLLECTION_NAME, collection.count())


def retrieve_legal_precedents(query: str, top_k: int = 3, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Retrieves relevant statutory citations, excerpt text, metadata, and relevance scores.

    Parameters:
      query   : Query string describing the pricing anomaly or legal infraction.
      top_k   : Number of relevant precedents to retrieve.
      db_path : Path to ChromaDB persistence directory.

    Returns:
      List[Dict[str, Any]] containing:
        - statute: Name of the statute (e.g. Essential Commodities Act 1955)
        - section: Section / clause reference (e.g. Section 3)
        - excerpt: Text excerpt of the statutory provision
        - title: Provision title
        - relevance_score: Float cosine similarity/relevance score between 0.0 and 1.0
        - id: Document ID
    """
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    embedding_func = _get_embedding_function()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
    )

    if collection.count() == 0:
        populate_database(db_path=db_path)

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    retrieved: List[Dict[str, Any]] = []
    for doc_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
        # Convert distance to similarity score in [0.0, 1.0]
        # For L2 or cosine distance, similarity can be computed as 1 / (1 + distance)
        similarity = float(round(1.0 / (1.0 + max(0.0, float(dist))), 4))
        retrieved.append({
            "id": doc_id,
            "statute": meta.get("statute", "Statutory Regulation"),
            "section": meta.get("section", ""),
            "title": meta.get("title", ""),
            "category": meta.get("category", ""),
            "excerpt": doc_text,
            "relevance_score": similarity,
        })

    return retrieved


def query_precedents(query_text: str, n_results: int = 2, db_path: str = DEFAULT_DB_PATH) -> List[str]:
    """
    Backward-compatible helper returning plain text strings of matching documents.
    """
    precedents = retrieve_legal_precedents(query=query_text, top_k=n_results, db_path=db_path)
    return [p["excerpt"] for p in precedents]


def main():
    populate_database()
    test_query = "Coordinated multi-vendor price collusion and artificial shortages"
    logger.info("Executing retrieval test for query: '%s'", test_query)
    results = retrieve_legal_precedents(test_query, top_k=3)
    for idx, res in enumerate(results):
        logger.info("[%d] %s (%s) - Score: %s", idx + 1, res["statute"], res["section"], res["relevance_score"])
        logger.info("    Excerpt: %s...", res["excerpt"][:120])


if __name__ == "__main__":
    main()
