# Low-RAM optimizations

- bounded CayleyPy caches for custom target states
- adapter cache trimming between rows
- optional compact/no profile mode in `search_improver_v3.py`
- chunked subprocess-based Colab runner to release memory after each chunk
- notebook defaults tuned for low RAM (`profile_mode=none`, low workers, beam-only, smaller chunk size)
