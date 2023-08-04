python sbu_cloudwriter.py \
  --path s3://initial-ai-datasets/image/downloaded/sbu-captions/ \
  --local ./cache/sbu \
  --remote s3://initial-ai-datasets/image/downloaded/sbu-captions-mds/ \
  --keep_parquet \
  --bucketed \
  --subfolder 1