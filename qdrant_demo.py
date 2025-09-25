from qdrant_client import AsyncQdrantClient, models
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import AsyncQdrantClient
import torch.nn.functional as F
from tools import is_text
import torch
from transformers import AutoTokenizer, AutoModel
import asyncio
import os
from itertools import count
from llama_index.core import VectorStoreIndex


REST_API_PORT=6333
EMBEDDINGS_DIM=768


# returns berta embeddings
def create_berta_embeddings(berta_model: AutoModel, 
                            berta_tokenizer: AutoTokenizer, 
                            inputs: list[str], 
                            batch_size=32, 
                            device: str='cuda',
                            prefix: str="search_document: "):
    def pool(hidden_state, mask, pooling_method="mean"):
        if pooling_method == "mean":
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif pooling_method == "cls":
            return hidden_state[:, 0]

    # add task prefix if exists:
    if prefix:
        inputs = [prefix + input_str for input_str in inputs]

    batch_count = (len(inputs) + batch_size - 1) // batch_size
    result_embeddings = []

    with torch.no_grad():
        for i in range(len(batch_count)):
            batch = inputs[i*batch_size: (i+1)*batch_size]
            tokenized_inputs = berta_tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors="pt").to(device)
            berta_model.to(device)
            outputs = berta_model(**tokenized_inputs)
            embeddings = pool(
                outputs.last_hidden_state, 
                tokenized_inputs["attention_mask"],
                pooling_method="mean"
            )

            embeddings = F.normalize(embeddings, p=2, dim=1).to('cpu')
            result_embeddings.append(embeddings)
    result_embeddings = torch.cat(result_embeddings, dim=0)

    return result_embeddings


def load_model(model_name:str="sergeyzh/BERTA"):
    print('loading model and tokenizer...')
    try:
        berta_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        berta_model = AutoModel.from_pretrained(model_name, local_files_only=True)
    except Exception as e:
        print('no local files found, load from server...')
        berta_tokenizer = AutoTokenizer.from_pretrained(model_name)
        berta_model = AutoModel.from_pretrained(model_name)

    return berta_model, berta_tokenizer
        

async def create_chunked_database(qdrant_client: AsyncQdrantClient,
                                  documents_dir: str='data/', 
                                  chunk_splitter:str='\n'*4, 
                                  collection:str='nsu_base'
                                  ):
    if not qdrant_client:
        return
    # check collection already created
    if await qdrant_client.collection_exists(collection):
        return
    await qdrant_client.create_collection(collection_name=collection,
                                          vectors_config=models.VectorParams(size=EMBEDDINGS_DIM, distance=models.Distance.COSINE)
                                          )
    berta_model, berta_tokenizer = load_model()
    id_counter = count(start=-1)    # global chunk id in qdrant database

    for file in filter(lambda f: is_text(f), os.listdir(documents_dir)):
        f_path = os.path.join(documents_dir, file)
        with open(f_path, encoding='utf-8') as f:
            data = f.read()

        # create embeddings of document's chunks:
        chunks = data.split(chunk_splitter)
        embeddings = create_berta_embeddings(berta_model, berta_tokenizer, chunks)
        local_id_counter = count(start=-1)  # chunk id inside document

        # insert embeddings to qdrant base:
        operation_info = await qdrant_client.upsert(collection_name=collection,
                                                    points=[models.PointStruct(id=next(id_counter),
                                                                               vector=embedding,
                                                                               payload={"doc_name": file, "chunk_id": next(local_id_counter)}
                                                                               )
                                                            for embedding in embeddings
                                                            ]
                                                    )
        if operation_info.status == models.UpdateStatus.ACKNOWLEDGED:
            print(f'WARNING: acknowledged request for doc - {file}')


# create async qdrant client
async def main():
    query_prefix = "search query: "
    base_name = 'nsu_base'
    qdrant_client = AsyncQdrantClient(url=f"http://localhost:{REST_API_PORT}")
    await create_chunked_database(qdrant_client)
    vector_store = QdrantVectorStore(aclient=qdrant_client, collection_name=base_name)

    # add prefix for query embeddings calculation:
    embed_model = HuggingFaceEmbedding(model_name="sergeyzh/BERTA", device='gpu', query_instruction=query_prefix)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, 
                                               embed_model=embed_model
                                               )
    query_engine = index.as_query_engine()
    query_engine.query("some query")


if __name__ == '__main__':
    asyncio.run(main())