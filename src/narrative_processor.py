"""
Process novels: chunking, narrative structuring, tagging.
Converts raw text → structured reasoning units.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class NarrativeProcessor:
    """Process novels into tagged chunks."""

    def process_all_novels(self, novels: Dict[str, str]) -> List[Dict]:
        """
        Process all novels and return structured chunks.
        
        Returns:
            List of {chunk_id, text, type, entities, timestamp_index}
        """
        chunks = []
        for book_name, full_text in novels.items():
            book_chunks = self._process_single_novel(book_name, full_text)
            chunks.extend(book_chunks)
        logger.info(f"Total chunks processed: {len(chunks)}")
        return chunks

    def _process_single_novel(self, book_name: str, full_text: str) -> List[Dict]:
        """Process a single novel."""
        # Placeholder: simple splitting by paragraphs
        paragraphs = full_text.split('\n\n')
        chunks = []

        for idx, para in enumerate(paragraphs):
            if len(para.strip()) < 50:  # Skip very short paragraphs
                continue

            chunks.append({
                'chunk_id': f"{book_name}_para_{idx}",
                'text': para,
                'type': 'NARRATIVE',  # TODO: use LLM to classify
                'entities': [],  # TODO: extract entities
                'timestamp_index': idx / len(paragraphs),  # Normalized position
            })

        return chunks
