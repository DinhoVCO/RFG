from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseVector
from kdir_src.models.all_models import load_models, get_docs_embeddings, get_and_save_docs_embeddings
from kdir_src.dataset.beir import get_sentences_corpus
from tqdm import tqdm
import math


client = QdrantClient(url="http://localhost:6333")

def get_points(all_embeddings,payloads):
    points = []
    for i, payload in tqdm(enumerate(payloads), desc= "creating points"):
        points.append(
            PointStruct(
                id=int(i), 
                payload=payload,
                vector={
                    "contriever":all_embeddings[0][i],
                    "contriever_ft":all_embeddings[1][i],
                    "dpr": all_embeddings[2][i],
                    "bge_large": all_embeddings[3][i],
                    "gte_large": all_embeddings[4][i],
                    "sparse_bm25": SparseVector(
                        indices=all_embeddings[5][i].indices,
                        values=all_embeddings[5][i].values
                    )
                }
            )
        )
    return points

def create_collection(client, collection_name, DIM_CONTRIEVER, DIM_CONTRIEVER_FT, DIM_DPR, DIM_BGE_L, DIM_GTE_L):
    if client.collection_exists(collection_name=collection_name):
        print(f"The collection '{collection_name}' already exists.")
        client.delete_collection(collection_name=collection_name)
    
    client.create_collection(
        collection_name= collection_name,
        vectors_config={
            "contriever": VectorParams(size=DIM_CONTRIEVER, distance=Distance.COSINE),
            "contriever_ft": VectorParams(size=DIM_CONTRIEVER_FT, distance=Distance.COSINE),
            "dpr": VectorParams(size=DIM_DPR, distance=Distance.COSINE),
            "bge_large": VectorParams(size=DIM_BGE_L, distance=Distance.COSINE),
            "gte_large": VectorParams(size=DIM_GTE_L, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse_bm25": SparseVectorParams(),
        }
    )

def insert_documents(collection_name, points, batch_size=100):
    total_batches = math.ceil(len(points) / batch_size)
    for i in tqdm(range(total_batches), desc="Inserting documents in batches"):
        batch = points[i * batch_size : (i + 1) * batch_size]
        client.upsert(collection_name=collection_name, points=batch)

def process_corpus_save_embedding(dataset_name, model_name, path, bz_emb= 32):
    models = load_models()
    sentences, payloads = get_sentences_corpus(dataset_name)
    get_and_save_docs_embeddings(dataset_name, models,model_name, path, sentences, bz_emb)

def process_corpus(dataset_name, bz_emb= 32, bz_qdrant = 100):
    models = load_models()
    sentences, payloads = get_sentences_corpus(dataset_name)
    all_embeddings = get_docs_embeddings(models, sentences, bz_emb)
    points = get_points(all_embeddings, payloads)


    DIM_CONTRIEVER = models['contriever'][2]
    DIM_CONTRIEVER_FT = models['contriever_ft'][2]
    DIM_DPR = models['context_dpr'][2]
    DIM_BGE_L = models['bge_large'][1]
    DIM_GTE_L = models['gte_large'][1]
    collection_name= f"kdir_{dataset_name}"
    create_collection(client, collection_name,  DIM_CONTRIEVER, DIM_CONTRIEVER_FT, DIM_DPR, DIM_BGE_L, DIM_GTE_L)
    insert_documents(collection_name, points, batch_size=bz_qdrant)
    print("Collection created")

def recuperar_documentos(collection_name, query_embeddings, vector_name, top_k=10):
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_embeddings,
        with_payload=True,
        using=vector_name,
        limit=top_k
    ).points
    return search_result

def recuperar_documentos_sparsos(collection_name, query_embeddings, vector_name, top_k=10):
    search_result = client.query_points(
        collection_name=collection_name,
        query=SparseVector(indices=query_embeddings.indices, values=query_embeddings.values),
        with_payload=True,
        using=vector_name,
        limit=top_k
    ).points
    return search_result

