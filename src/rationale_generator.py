"""
Generate human-readable evidence rationale.
Extract key supporting/contradicting passages and explain linkage.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class RationealeGenerator:
    """Generate evidence-based explanations."""

    def generate(
        self,
        example_id: int,
        final_decision: Dict,
        claims: List[Dict],
        claim_scores: Dict,
        claim_evidence: Dict,
        character_name: str,
        backstory_content: str,
    ) -> str:
        """
        Generate human-readable rationale.
        
        Returns:
            Explanation string (1-3 sentences)
        """
        logger.info(f"Generating rationale for example {example_id}...")

        # Find most decisive claims
        decisive_claims = sorted(
            claim_scores.items(),
            key=lambda x: abs(x[1]['aggregate_score']),
            reverse=True,
        )[:2]  # Top 2 most decisive

        rationale_parts = []

        for claim_id, score in decisive_claims:
            claim_text = next(c['text'] for c in claims if c['id'] == claim_id)
            evidence = claim_evidence.get(claim_id, [])

            if score['contradict_count'] > 0:
                if evidence:
                    passage_snippet = evidence[0]['text'][:100]
                    rationale_parts.append(
                        f"Backstory claim '{claim_text}' contradicted by novel passage: "
                        f"'{passage_snippet}...'"
                    )
            elif score['support_count'] > 0:
                if evidence:
                    passage_snippet = evidence[0]['text'][:100]
                    rationale_parts.append(
                        f"Backstory claim '{claim_text}' supported by novel passage: "
                        f"'{passage_snippet}...'"
                    )

        # Combine
        if rationale_parts:
            rationale = ". ".join(rationale_parts) + "."
        else:
            rationale = "Insufficient evidence to determine consistency."

        return rationale