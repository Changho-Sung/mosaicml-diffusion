composer -m scripts.precompute_latents_sbu \
  --remote_download s3://initial-ai-datasets/image/processed/sbu-captions-mds/256-512 \
  --local ~/cache/sbu-mds/256-512 \
  --remote_upload s3://initial-ai-datasets/image/processed/sbu-captions-precompute \
  --bucket 3
