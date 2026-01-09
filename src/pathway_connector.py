"""
Pathway framework connector for vector store and text indexing.
Handles ingestion, chunking, and multi-passage retrieval.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    PATHWAY_CONFIG, CHUNK_CONFIG, EMBEDDING_MODEL,
    PROCESSED_CHUNKS_PATH, EMBEDDINGS_CACHE_PATH
)

logger = logging.getLogger(__name__)


class PathwayConnector:
    """
    Pathway framework for vector store and indexing.
    In production, integrates with Pathway's Python API.
    For this implementation, uses SentenceTransformers + FAISS as surrogate.
    """

    def __init__(self):
        """Initialize embedder and vector store."""
        logger.info(f"Initializing Pathway connector with {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.chunks = []  # List of {text, chunk_id, embedding, metadata}
        self.chunk_texts = []
        self.chunk_embeddings = []
        self.chunk_metadata = []

    def ingest_novels(self, novels: Dict[str, str]) -> None:
        """
        Ingest novel texts, create chunks, and build vector store.
        
        Args:
            novels: Dict[book_name → full_text]
        """
        logger.info(f"Ingesting {len(novels)} novels into Pathway vector store...")

        for book_name, full_text in novels.items():
            self._process_novel(book_name, full_text)

        # Compute embeddings for all chunks
        logger.info(f"Computing embeddings for {len(self.chunk_texts)} chunks...")
        self.chunk_embeddings = self.embedder.encode(
            self.chunk_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        logger.info(f"Ingestion complete: {len(self.chunks)} chunks")

    def _process_novel(self, book_name: str, full_text: str) -> None:
        """
        Process a single novel: chunking with metadata extraction.
        
        Args:
            book_name: Name of the novel
            full_text: Complete novel text
        """
        chunks = self._create_chunks(full_text)

        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_id = f"{book_name.lower().replace(' ', '_')}_chunk_{chunk_idx}"
            metadata = {
                'book_name': book_name,
                'chunk_idx': chunk_idx,
                'char_offset_start': full_text.find(chunk_text),
                'text_preview': chunk_text[:100],
            }

            self.chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'metadata': metadata,
            })
            self.chunk_texts.append(chunk_text)
            self.chunk_metadata.append(metadata)

    def _create_chunks(self, text: str, chunk_size: int = 512, overlap: int = 256) -> List[str]:
        """
        Create overlapping chunks from text using simple token-based splitting.
        
        Args:
            text: Full text
            chunk_size: Tokens per chunk
            overlap: Overlap in tokens
            
        Returns:
            List of chunk texts
        """
        # Simple word-based chunking
        words = text.split()
        chunks = []
        
        chunk_word_size = max(1, chunk_size // 5)  # rough estimate
        overlap_words = max(1, overlap // 5)

        for i in range(0, len(words), chunk_word_size - overlap_words):
            chunk_words = words[i : i + chunk_word_size]
            if chunk_words:
                chunks.append(' '.join(chunk_words))

        return chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        character_filter: Optional[str] = None,
        book_filter: Optional[str] = None,
        recency_bias: float = 0.0,
    ) -> List[Dict]:
        """
        Retrieve top-k most similar chunks to query.
        
        Args:
            query: Search query text
            top_k: Number of results
            character_filter: Optional character name to filter by
            book_filter: Optional book name to filter by
            recency_bias: Weight toward later chunks (0.0 to 1.0)
            
        Returns:
            List of (chunk_text, metadata, score) tuples
        """
        # Embed query
        query_embedding = self.embedder.encode(query, convert_to_numpy=True)

        # Compute similarity to all chunks
        similarities = np.dot(self.chunk_embeddings, query_embedding)

        # Apply filters
        valid_indices = list(range(len(self.chunks)))

        if book_filter:
            valid_indices = [
                i for i in valid_indices
                if self.chunk_metadata[i]['book_name'] == book_filter
            ]

        # Sort by similarity
        sorted_indices = sorted(
            valid_indices,
            key=lambda i: similarities[i],
            reverse=True,
        )

        # Return top-k
        results = []
        for idx in sorted_indices[:top_k]:
            results.append({
                'chunk_id': self.chunks[idx]['id'],
                'text': self.chunks[idx]['text'],
                'metadata': self.chunks[idx]['metadata'],
                'score': float(similarities[idx]),
            })

        return results

    def save(self, output_path: Path) -> None:
        """Save vector store to disk."""
        data = {
            'chunks': self.chunks,
            'chunk_texts': self.chunk_texts,
            'chunk_embeddings': self.chunk_embeddings.tolist(),  # JSON-serializable
            'chunk_metadata': self.chunk_metadata,
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved Pathway vector store to {output_path}")

    def load(self, input_path: Path) -> None:
        """Load vector store from disk."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        self.chunks = data['chunks']
        self.chunk_texts = data['chunk_texts']
        self.chunk_embeddings = np.array(data['chunk_embeddings'])
        self.chunk_metadata = data['chunk_metadata']
        logger.info(f"Loaded Pathway vector store from {input_path}")
