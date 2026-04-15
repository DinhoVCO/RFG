import numpy as np
from kdir_src.models.all_models import get_query_embeddings
from qdrant_client.models import SparseVector
from kdir_src.utils.qdrant import recuperar_documentos, recuperar_documentos_sparsos
from kdir_src.dataset.beir import get_sentences_queries
from kdir_src.utils.inference import get_inference_results
from tqdm import tqdm
from kdir_src.models.all_models import get_query_embeddings_by_model, get_query_embeddings

import os
import json
import torch
import time

#FUSION_PROMPT = """You are an expert AI assistant in reformulating questions to improve search results in a database.
#Analyze the following "Query" Then, generate a "New Query" that explores the original question from a different or more specific perspective.
#Query: {}
#New Query:"""

#FUSION_PROMPT = """You are an expert AI assistant in reformulating questions.
#Analyze the following "Query" to understand its core intent, context, and the type of information sought. Then, generate a "New Query" that is more specific, detailed, or reframed to yield a better or more precise answer. The new query should maintain the original's intent.

#IMPORTANT: Your response must contain ONLY the new query itself and nothing else. Do not include explanations, preambles, or the label "New Query:".

#Query: {}
#New Query:"""

FUSION_PROMPT = """You are an expert AI assistant in reformulating questions.
Analyze the following "Query" to understand its core intent, context, and the type of information sought. Then, generate a different "New Query". The new query should maintain the original's intent.
IMPORTANT: Your response must contain ONLY the new query itself and nothing else. Do not include explanations, preambles, or the label "New Query:".
Query: {}
New Query:"""

class Promptor:
    def __init__(self, task: str):
        self.task = task
    
    def build_prompt(self, query: str):
        return FUSION_PROMPT.format(query)


class RAG_fusion:
    def __init__(self, datset_name, promptor, generator, encoder_models):
        self.promptor = promptor
        self.generator = generator
        self.encoder_models = encoder_models
        self.dataset_name = datset_name
    
    def prompt(self, query):
        return self.promptor.build_prompt(query)

    def generate(self, query, max_retries=5, retry_delay_seconds=1):
        prompt = self.promptor.build_prompt(query)
        for attempt in range(max_retries):
            try:
                hypothesis_querys = self.generator.generate(prompt) 
                return hypothesis_querys
            except Exception as e: # Captura excepciones generales. Puedes ser más específico si conoces los errores de tu API.
                print(f"Error generating for query '{query}' (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay_seconds} seconds...")
                    time.sleep(retry_delay_seconds)
                else:
                    print(f"Max retries reached for query '{query}'. Giving up.")
                    raise # Vuelve a lanzar la excepción si se agotaron los reintentos


    def save_generated_docs(self, path='../doc_gen/rag_fusion/'):
        os.makedirs(path, exist_ok=True) # Ensure the directory exists
        sentences, queries_ids = get_sentences_queries(self.dataset_name)
        output_filepath = os.path.join(path, f"generated_documents_{self.dataset_name}.jsonl")
        existing_query_ids = set()
        if os.path.exists(output_filepath):
            with open(output_filepath, 'r', encoding='utf-8') as f_read:
                for line in f_read:
                    try:
                        entry = json.loads(line)
                        if "query_id" in entry:
                            existing_query_ids.add(entry["query_id"])
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON line in {output_filepath}: {line.strip()}")
                        continue
        
        print(f"Found {len(existing_query_ids)} previously generated documents in '{output_filepath}'.")
        with open(output_filepath, 'a', encoding='utf-8') as f_write:
            for i, query in tqdm(enumerate(sentences), desc='Processing queries (generate or skip)'):
                current_query_id = queries_ids[i]
                if current_query_id in existing_query_ids:
                    continue
                hypothesis_querys = self.generate(query)
                entry = {
                    "query_id": current_query_id,
                    "query": query,
                    "generated_documents": hypothesis_querys
                }
                f_write.write(json.dumps(entry, ensure_ascii=False) + '\n')        
        print(f"Document generation complete. Results saved in: {output_filepath}")

    def encode(self, hypothesis_documents):
        c_emb_list = get_query_embeddings(self.encoder_models, hypothesis_documents,32,False)
        return c_emb_list

    def get_all_querys_embeddings_from_jsonl(self, datos):
        queries_ids=[]
        contriever=[]
        contriever_ft=[]
        dpr=[]
        bge_large=[]
        gte_large=[]
        sparse_bm25=[]       
        for doc in tqdm(datos, desc='getting embeddings of all generated docs'):
            queries_ids.append(doc['query_id'])
            hypothesis_documents = doc['generated_documents']
            hyde_vector = self.encode(hypothesis_documents)
            contriever.append(hyde_vector[0])
            contriever_ft.append(hyde_vector[1])
            dpr.append(hyde_vector[2])
            bge_large.append(hyde_vector[3])
            gte_large.append(hyde_vector[4])
            sparse_bm25.append(hyde_vector[5])
        all_querys_embeddings=[contriever,contriever_ft, dpr, bge_large, gte_large, sparse_bm25]
        return all_querys_embeddings, queries_ids
    
    def reciprocal_rank_fusion_from_scores(self, results: list[dict], k=60) -> dict:
        """
        Realiza Reciprocal Rank Fusion y devuelve un diccionario ordenado
        con los resultados {id_documento: puntaje_rrf_final}.
        """
        fused_scores = {}
        for result_dict in results:
            ranked_list = [
                doc_id for doc_id, score in sorted(result_dict.items(), key=lambda item: item[1], reverse=True)
            ]
            for rank, doc_id in enumerate(ranked_list):
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (rank + k)
        sorted_items = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        reranked_dict = dict(sorted_items)
        return reranked_dict

    def rrf_all_results(self, queries_ids,all_results):
        result_final = {}
        for ids in queries_ids:
            query_results=[resultado[ids] for resultado in all_results]
            result_final[ids] = self.reciprocal_rank_fusion_from_scores(query_results)
        return result_final


    def get_results_from_jsonl_rag_fusion(self, path_doc_gen, path='../results/',batch_size=32, top_k=10):
        os.makedirs(path, exist_ok=True)
        datos = []
        with open(path_doc_gen, 'r', encoding='utf-8') as f:
            for linea in f:
                datos.append(json.loads(linea.strip()))
        all_sentence_embeddings, queries_ids = self.get_all_querys_embeddings_from_jsonl(datos)

        
        collection_name= f"kdir_{self.dataset_name}"
        vectors =["contriever","contriever_ft","dpr","bge_large","gte_large","sparse_bm25"]
        i=0
        for model_embeddings in all_sentence_embeddings:
            listas_transpuestas = [list(item) for item in zip(*model_embeddings)]
            all_results=[]
            for lista in listas_transpuestas:
                results_partial = get_inference_results(lista, queries_ids, collection_name, vectors[i], top_k)
                all_results.append(results_partial)
            results = self.rrf_all_results(queries_ids, all_results)             

            with open(path+f"results_{vectors[i]}.json", "w") as f:
                json.dump(results, f, indent=2)
            i+=1
