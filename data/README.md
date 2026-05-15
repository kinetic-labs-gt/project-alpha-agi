# Alpha-AGI Data Specification

This directory holds the blueprints for data ingestion during Phase 2 pre-training.

## Manifest Format
Incoming raw datasets must strictly conform to the `manifest.schema.json` specification to guarantee a reliable data pipeline.

### Required Fields
- `text`: The sequence content used for training.
- `source`: The ingestion source.
- `lang`: The language.
- `dedup_hash`: A cryptographic signature representing the text, strictly evaluated to eliminate duplication during preprocessing.

### Optional Fields
- `license`: Rights assignment.
- `quality_score`: Floating point score used downstream for data curriculum filtering.
