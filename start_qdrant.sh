docker run \
	--rm \
	--gpus=all \
	-p 6333:6333 \
	-p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
	-e QDRANT__GPU__INDEXING=1 \
	qdrant/qdrant:gpu-nvidia-latest