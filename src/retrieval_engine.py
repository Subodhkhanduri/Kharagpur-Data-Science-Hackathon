"""
Constraint-aware retrieval: fetch relevant passages from novel
that either support or contradict each backstory claim.
"""

import logging
from typing import List, Dict, Optional
from config import RETRIEVAL_CONFIG

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Retrieve evidence passages for claims."""

    def __init__(self, chunks_data: List[Dict], pathway_connector, book_name: str, character_name: str):
        """
        Initialize retrieval engine.
        
        Args:
            chunks_data: Pre-processed chunks with metadata
            pathway_connector: Pathway vector store
            book_name: Novel name (for filtering)
            character_name: Character name (for context)
        """
        self.chunks_data = chunks_data
        self.pathway = pathway_connector
        self.book_name = book_name
        self.character_name = character_name

    def retrieve_evidence(self, claim: Dict) -> List[Dict]:
        """
        Retrieve multiple disjoint passages relevant to a claim.
        
        Args:
            claim: Claim object with 'text' and 'evidence_keywords'
            
        Returns:
            List of evidence passages with scores and metadata
        """
        logger.info(f"Retrieving evidence for claim: {claim['text'][:50]}...")

        # Extract search keywords from claim
        keywords = claim.get('evidence_keywords', [])
        
        # Build composite query
        query = f"{claim['text']} {' '.join(keywords)}"

        # Retrieve from Pathway
        retrieved = self.pathway.retrieve(
            query=query,
            top_k=RETRIEVAL_CONFIG['top_k'],
            book_filter=self.book_name,
        )

        logger.info(f"Retrieved {len(retrieved)} passages")
        return retrieved
