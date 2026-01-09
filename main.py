import logging
import pandas as pd
from pathlib import Path
from config import (
    TRAIN_CSV_PATH, NOVELS_DIR, RESULTS_CSV_PATH,
    PROCESSED_CHUNKS_PATH, PROCESSED_CLAIMS_PATH, LOGGING_CONFIG
)

from src.narrative_processor import NarrativeProcessor
from src.claim_decomposer import ClaimDecomposer
from src.pathway_connector import PathwayConnector
from src.retrieval_engine import RetrievalEngine
from src.reasoning_engine import ReasoningEngine
from src.aggregation_engine import AggregationEngine
from src.rationale_generator import RationealeGenerator

# Setup logging
logging.basicConfig(
    level=LOGGING_CONFIG["level"],
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["log_file"]),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main execution pipeline."""
    logger.info("=" * 80)
    logger.info("KHARAGPUR DATA SCIENCE HACKATHON 2026 - NARRATIVE CONSISTENCY SYSTEM")
    logger.info("=" * 80)

    # STEP 0: LOAD INPUT DATA
    logger.info("\n[STEP 0] Loading input data...")
    
    # Load CSV
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    logger.info(f"Loaded {len(df_train)} examples from train-1.csv")
    
    # Load novels
    novels = {}
    for book_name, novel_path in {
        "The Count of Monte Cristo": Path(NOVELS_DIR) / "The-Count-of-Monte-Cristo.txt",
        "In Search of the Castaways": Path(NOVELS_DIR) / "In-search-of-the-castaways.txt",
    }.items():
        with open(novel_path, "r", encoding="utf-8") as f:
            novels[book_name] = f.read()
        logger.info(f"Loaded {book_name} ({len(novels[book_name])} chars)")


    # STEP 1: PATHWAY INGESTION & INDEXING
    logger.info("\n[STEP 1] Initializing Pathway connector and ingesting novels...")
    
    pathway_connector = PathwayConnector()
    pathway_connector.ingest_novels(novels)
    logger.info("Novels ingested into Pathway vector store")


    # STEP 2: NARRATIVE PROCESSING & CHUNKING
    logger.info("\n[STEP 2] Processing narratives (chunking, tagging)...")
    
    processor = NarrativeProcessor()
    chunks_data = processor.process_all_novels(novels)
    logger.info(f"Created {len(chunks_data)} chunks across all novels")


    # STEP 3: PROCESS EACH EXAMPLE
    logger.info("\n[STEP 3] Processing examples...")
    
    results = []

    for idx, row in df_train.iterrows():
        if idx % 10 == 0:
            logger.info(f"Processing example {idx + 1} / {len(df_train)}")

        example_id = row['id']
        book_name = row['book_name']
        character = row['char']
        backstory_content = row['content']
        true_label = row['label']

        try:
            # 3a. Decompose backstory into claims
            decomposer = ClaimDecomposer()
            claims = decomposer.decompose(
                backstory_text=backstory_content,
                character_name=character,
                book_name=book_name,
            )

            # 3b. Retrieve relevant evidence for each claim
            retriever = RetrievalEngine(
                chunks_data=chunks_data,
                pathway_connector=pathway_connector,
                book_name=book_name,
                character_name=character,
            )
            
            claim_evidence = {}
            for claim in claims:
                evidence = retriever.retrieve_evidence(claim)
                claim_evidence[claim['id']] = evidence


            # 3c. Score each claim against evidence

            reasoner = ReasoningEngine()
            claim_scores = {}
            for claim in claims:
                score_result = reasoner.score_claim(
                    claim=claim,
                    evidence_passages=claim_evidence[claim['id']],
                    character_name=character,
                )
                claim_scores[claim['id']] = score_result

 
            # 3d. Aggregate scores → final decision
 
            aggregator = AggregationEngine()
            final_decision = aggregator.aggregate(claim_scores)

 
            # 3e. Generate evidence rationale (optional, Track A bonus)
 
            generator = RationealeGenerator()
            rationale = generator.generate(
                example_id=example_id,
                final_decision=final_decision,
                claims=claims,
                claim_scores=claim_scores,
                claim_evidence=claim_evidence,
                character_name=character,
                backstory_content=backstory_content,
            )

 
            # Store result
            results.append({
                'story_id': example_id,
                'book_name': book_name,
                'character': character,
                'prediction': final_decision['label'],  # 0 or 1
                'confidence': final_decision['confidence'],
                'true_label': 1 if true_label == 'consistent' else 0,
                'rationale': rationale,
            })

        except Exception as e:
            logger.error(f"Error processing example {example_id}: {str(e)}")
            results.append({
                'story_id': example_id,
                'book_name': book_name,
                'character': character,
                'prediction': 0,  # default to contradiction on error
                'confidence': 0.0,
                'true_label': 1 if true_label == 'consistent' else 0,
                'rationale': f"Error: {str(e)}",
            })


    # STEP 4: EVALUATE & EXPORT RESULTS
    logger.info("\n[STEP 4] Exporting results...")

    results_df = pd.DataFrame(results)
    
    # Export predictions CSV (required format)
    output_df = results_df[['story_id', 'prediction', 'rationale']].copy()
    output_df.to_csv(RESULTS_CSV_PATH, index=False)
    logger.info(f"Results exported to {RESULTS_CSV_PATH}")

    # Calculate accuracy for reference
    accuracy = (results_df['prediction'] == results_df['true_label']).mean()
    logger.info(f"\nAccuracy on {len(results_df)} examples: {accuracy:.4f}")
    logger.info(f"Prediction distribution:\n{results_df['prediction'].value_counts()}")

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    return results_df


if __name__ == "__main__":
    results = main()
