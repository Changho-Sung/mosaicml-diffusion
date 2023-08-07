python sbu_cloudwriter.py \
  --path s3://initial-ai-datasets/image/downloaded/sbu-captions/ \
  --local ~/cache/sbu \
  --remote s3://initial-ai-datasets/image/processed/sbu-captions-mds/ \
  --keep_parquet \
  --keep_cache \
  --bucketed \
  --subfolder 3
