"""
Decompose backstory into atomic, testable claims.
Each claim is a self-contained hypothesis about the character.
"""

import logging
import json
from typing import List, Dict
from config import LLM_CONFIG

logger = logging.getLogger(__name__)


class ClaimDecomposer:
    """Extract atomic claims from backstory text."""

    def __init__(self):
        """Initialize LLM for claim decomposition."""
        # For this implementation, use OpenAI/DeepSeek API
        pass

    def decompose(
        self,
        backstory_text: str,
        character_name: str,
        book_name: str,
    ) -> List[Dict]:
        """
        Decompose backstory into atomic claims.
        
        Args:
            backstory_text: Full backstory narrative
            character_name: Target character name
            book_name: Novel name (for context)
            
        Returns:
            List of claims, each with: id, text, type, confidence, evidence_keywords
        """
        logger.info(f"Decomposing backstory for {character_name} in {book_name}")

        # Call LLM to extract claims
        prompt = self._build_prompt(backstory_text, character_name, book_name)
        
        response_text = self._call_llm(prompt)
        claims = self._parse_claims(response_text)

        logger.info(f"Extracted {len(claims)} claims")
        return claims

    def _build_prompt(self, backstory: str, character_name: str, book_name: str) -> str:
        """
        Build prompt for LLM to decompose backstory.
        """
        return f"""You are an expert literary analyst. Your task is to decompose a character backstory into atomic, testable claims.

Character: {character_name}
Novel: {book_name}

BACKSTORY:
{backstory}

Please extract ALL claims about this character's past, beliefs, experiences, and assumptions.

For each claim:
1. State it as a simple, testable assertion (e.g., "fears authority", "lost parent young")
2. Categorize it as one of: BACKGROUND | TRAIT | BELIEF | EXPERIENCE | ASSUMPTION | RELATIONSHIP
3. Rate your confidence in the clarity/specificity of the claim (0.0 to 1.0)
4. List 3-5 key words that someone reading the novel might use to verify this claim

Format your response as a JSON array:
[
  {{
    "claim_text": "...",
    "claim_type": "...",
    "confidence": 0.8,
    "evidence_keywords": ["word1", "word2", ...]
  }},
  ...
]

Return ONLY the JSON array, no other text."""

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API (DeepSeek, OpenAI, or Claude).
        """
        # Placeholder: integrate with actual API
        # For now, return empty JSON to allow pipeline to continue
        logger.warning("LLM call not yet implemented; using placeholder")
        return "[]"

    def _parse_claims(self, response_text: str) -> List[Dict]:
        """
        Parse LLM response into claim objects.
        """
        try:
            claims_list = json.loads(response_text)
            # Add IDs
            for i, claim in enumerate(claims_list):
                claim['id'] = f"claim_{i}"
            return claims_list
        except json.JSONDecodeError:
            logger.error(f"Failed to parse claims from LLM response")
            return []
