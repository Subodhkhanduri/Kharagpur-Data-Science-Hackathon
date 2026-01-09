"""
Aggregate per-claim scores into a final binary decision:
Consistent (1) or Contradictory (0).
"""

import logging
from typing import Dict, List
from config import AGGREGATION_CONFIG

logger = logging.getLogger(__name__)


class AggregationEngine:
    """Aggregate claim scores into final decision."""

    def __init__(self):
        """Initialize aggregation rules."""
        self.config = AGGREGATION_CONFIG

    def aggregate(self, claim_scores: Dict[str, Dict]) -> Dict:
        """
        Aggregate per-claim scores → final binary decision.
        
        Rules:
        1. If ANY claim has strong contradiction → default to 0 (Contradict)
        2. Weight contradictions 2x stronger than supports
        3. Confidence-weighted voting
        4. If aggregate_score > threshold → 1 (Consistent), else 0
        
        Args:
            claim_scores: Dict[claim_id → {aggregate_score, confidence, ...}]
            
        Returns:
            {label: 0 or 1, confidence: float, reasoning: str}
        """
        logger.info(f"Aggregating {len(claim_scores)} claim scores...")

        if not claim_scores:
            logger.warning("No claim scores to aggregate")
            return {'label': 0, 'confidence': 0.0, 'reasoning': 'No evidence'}

        # Rule 1: Check for strong contradictions
        for claim_id, score in claim_scores.items():
            if score['contradict_count'] > 0 and score['aggregate_score'] < -0.5:
                logger.info(f"Strong contradiction found in claim {claim_id}")
                return {
                    'label': 0,
                    'confidence': 0.9,
                    'reasoning': f'Causal contradiction in claim: {claim_id}',
                }

        # Rule 2-3: Confidence-weighted aggregation
        weighted_scores = []
        for claim_id, score in claim_scores.items():
            confidence = max(score['confidence'], 1e-6)
            
            # Apply weights: contradiction negative, support positive
            weighted_score = (
                score['support_count'] * self.config['support_weight'] -
                score['contradict_count'] * self.config['contradiction_weight']
            ) / max(score['support_count'] + score['contradict_count'], 1)
            
            weighted_scores.append(weighted_score * confidence)

        # Rule 4: Average + threshold
        avg_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0
        final_label = 1 if avg_score > self.config['decision_threshold'] else 0

        confidence = min(abs(avg_score), 1.0)  # Clip to [0, 1]

        logger.info(f"Final decision: {final_label} (score: {avg_score:.4f}, confidence: {confidence:.4f})")

        return {
            'label': final_label,
            'confidence': confidence,
            'reasoning': f'Aggregated score: {avg_score:.4f}',
        }