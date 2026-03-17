from kdir_src.dataset.beir import get_sentences_queries, get_sentences_queries_openai
from kdir_src.models.all_models import get_query_embeddings_by_model, get_query_embeddings
from kdir_src.utils.qdrant import recuperar_documentos, recuperar_documentos_sparsos
from kdir_src.utils.inference import get_inference_results
from tqdm import tqdm 
import torch
import numpy as np
import json
import os
import time

class Fire:
    def __init__(self, dataset_name, promptor, generator, encoder_models):
        self.promptor = promptor
        self.generator = generator
        self.encoder_models = encoder_models
        self.dataset_name = dataset_name
    
    def prompt(self, query):
        return self.promptor.build_prompt(query)

    def _is_rate_limit_error(self, e: Exception) -> bool:
        """Detecta si el error es por rate limit (429)."""
        err_str = str(e).lower()
        return "429" in err_str or "rate_limit" in err_str or "rate_limited" in err_str

    def generate(self, prompt, max_retries=5, retry_delay_seconds=1, rate_limit_delay_seconds=60):
        for attempt in range(max_retries):
            try:
                hypothesis_documents = self.generator.generate(prompt) 
                return hypothesis_documents
            except Exception as e: 
                print(f"Error generating for query '' (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    delay = rate_limit_delay_seconds if self._is_rate_limit_error(e) else retry_delay_seconds
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Max retries reached for query ''. Giving up.")
                    raise

    def get_rel_docs_by_model(self, model_embeddings, collection_name, vector_name, top_k=5):
        docs = []
        for i, query_embedding in tqdm(enumerate(model_embeddings), desc=f"obteniendo rel_docs {vector_name}"):
            processed_query_embedding = query_embedding 
            if isinstance(query_embedding, torch.Tensor):
                processed_query_embedding = query_embedding.tolist()
            elif isinstance(query_embedding, np.ndarray):
                 processed_query_embedding = query_embedding.tolist()
            if(vector_name!="sparse_bm25"):
                rel_docs = recuperar_documentos(collection_name, processed_query_embedding, vector_name, top_k)
            else:
                rel_docs = recuperar_documentos_sparsos(collection_name, processed_query_embedding, vector_name, top_k)
            docs.append(rel_docs)
        return docs


    def get_all_rel_docs(self, batch_size=32, top_k=5):
        sentences, queries_ids = get_sentences_queries(self.dataset_name)
        collection_name= f"kdir_{self.dataset_name}"
        query_emb_list = get_query_embeddings(self.encoder_models, sentences , batch_size ,True)
        vectors =["contriever","contriever_ft","dpr","bge_large","gte_large","sparse_bm25"]
        all_rel_docs=[]
        for i, model_embeddings in enumerate(query_emb_list):
            rel_docs_per_model = self.get_rel_docs_by_model(model_embeddings, collection_name, vectors[i], top_k)
            all_rel_docs.append(rel_docs_per_model)
        return sentences, queries_ids, all_rel_docs


    def generate_and_save_prf(self, path='../doc_gen/fire/', batch_size=32, top_k=5):
        vectors =["contriever","contriever_ft","dpr","bge_large","gte_large","sparse_bm25"]
        sentences, queries_ids, all_rel_docs = self.get_all_rel_docs(batch_size, top_k)
        for j, rel_docs in enumerate(all_rel_docs):
            os.makedirs(path+f'{vectors[j]}/', exist_ok=True)
            output_filepath = os.path.join(path+f'{vectors[j]}/', f"generated_documents_{self.dataset_name}.jsonl")
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
                    prompts = self.promptor.get_prompt(query, rel_docs[i])
                    ans1 = self.generate(prompts[0])
                    ans2 = self.generate(prompts[1])
                    ans3 = self.generate(prompts[2])
                    entry = {
                        "query_id": current_query_id,
                        "query": query,
                        "generated_document": ans1,
                        "generated_query": ans2,
                        "generated_answer": ans3
                    }
                    f_write.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"Document generation complete. Results saved in: {output_filepath}")

    def get_datos_fire_per_dataset(self, path_doc_gen):
        vectors =["contriever","contriever_ft","dpr","bge_large","gte_large","sparse_bm25"]
        datos_por_vector = {}
        for vect in vectors:
            datos = []
            with open(path_doc_gen+f'{vect}/generated_documents_{self.dataset_name}.jsonl', 'r', encoding='utf-8') as f:
                for linea in f:
                    datos.append(json.loads(linea.strip()))
                datos_por_vector[vect]=datos
        return datos_por_vector



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

    def rrf_all_results(self, queries_ids, results_query,results_doc, results_new_query, results_answer):
        result_final = {}
        for ids in queries_ids:
            result_final[ids] = self.reciprocal_rank_fusion_from_scores([results_query[ids], results_doc[ids], results_new_query[ids], results_answer[ids]])
        return result_final

    def rrf_doc_answer_results(self, queries_ids, results_doc, results_answer):
        result_final = {}
        for ids in queries_ids:
            result_final[ids] = self.reciprocal_rank_fusion_from_scores([results_doc[ids], results_answer[ids]])
        return result_final

    def rrf_doc_answer_query_results(self, queries_ids,results_query, results_doc, results_answer):
        result_final = {}
        for ids in queries_ids:
            result_final[ids] = self.reciprocal_rank_fusion_from_scores([results_query[ids], results_doc[ids], results_answer[ids]])
        return result_final

    def get_3best_results_rrf_from_jsonl_fire(self, path_doc_gen, path='../results/fire/doc5/',batch_size=32, top_k=10, top_k_final=10):
        os.makedirs(path, exist_ok=True)
        datos_por_vector = self.get_datos_fire_per_dataset(path_doc_gen)
        collection_name= f"kdir_{self.dataset_name}"
        for model in datos_por_vector:
            query_list=[]
            sentences_doc = []
            sentences_answer = []
            queries_ids = []
            for doc in datos_por_vector[model]:
                query_list.append(doc['query'])
                sentences_doc.append(doc['generated_document'][0])
                sentences_answer.append(doc['generated_answer'][0])
                queries_ids.append(doc['query_id'])

            embeddings_query_list = get_query_embeddings_by_model(self.encoder_models,model, query_list ,128,True)
            embeddings_doc = get_query_embeddings_by_model(self.encoder_models,model, sentences_doc ,128,True)
            embeddings_answer = get_query_embeddings_by_model(self.encoder_models,model, sentences_answer ,128,True)
            
            results_query = get_inference_results(embeddings_query_list, queries_ids, collection_name, model, top_k)
            results_doc = get_inference_results(embeddings_doc, queries_ids, collection_name, model, top_k)
            results_answer = get_inference_results(embeddings_answer, queries_ids, collection_name, model, top_k)
            results = self.rrf_doc_answer_query_results(queries_ids, results_query,results_doc, results_answer)             
            with open(os.path.join(path, f"results_{model}.json"), "w") as f:
                    json.dump(results, f, indent=2)
            
            
    def get_all_results_rrf_from_jsonl_fire(self, path_doc_gen, path='../results/fire/doc5/',batch_size=32, top_k=10, top_k_final=10):
        os.makedirs(path, exist_ok=True)
        datos_por_vector = self.get_datos_fire_per_dataset(path_doc_gen)
        collection_name= f"kdir_{self.dataset_name}"
        for model in datos_por_vector:
            query_list=[]
            sentences_doc = []
            sentences_query = []
            sentences_answer = []
            queries_ids = []
            for doc in datos_por_vector[model]:
                query_list.append(doc['query'])
                sentences_doc.append(doc['generated_document'][0])
                sentences_query.append(doc['generated_query'][0])
                sentences_answer.append(doc['generated_answer'][0])
                queries_ids.append(doc['query_id'])

            embeddings_query_list = get_query_embeddings_by_model(self.encoder_models,model, query_list ,128,True)
            embeddings_doc = get_query_embeddings_by_model(self.encoder_models,model, sentences_doc ,128,True)
            embeddings_new_query = get_query_embeddings_by_model(self.encoder_models,model, sentences_query ,128,True)
            embeddings_answer = get_query_embeddings_by_model(self.encoder_models,model, sentences_answer ,128,True)
            
            results_query = get_inference_results(embeddings_query_list, queries_ids, collection_name, model, top_k)
            results_doc = get_inference_results(embeddings_doc, queries_ids, collection_name, model, top_k)
            results_new_query = get_inference_results(embeddings_new_query, queries_ids, collection_name, model, top_k)
            results_answer = get_inference_results(embeddings_answer, queries_ids, collection_name, model, top_k)
            results = self.rrf_all_results(queries_ids, results_query,results_doc, results_new_query, results_answer)             
            with open(os.path.join(path, f"results_{model}.json"), "w") as f:
                    json.dump(results, f, indent=2)
                
    def get_doc_answer_results_rrf_from_jsonl_fire(self, path_doc_gen, path='../results/fire/doc5/',batch_size=32, top_k=10, top_k_final=10):
        os.makedirs(path, exist_ok=True)
        datos_por_vector = self.get_datos_fire_per_dataset(path_doc_gen)
        collection_name= f"kdir_{self.dataset_name}"
        for model in datos_por_vector:
            sentences_doc = []
            sentences_answer = []
            queries_ids = []
            for doc in datos_por_vector[model]:
                sentences_doc.append(doc['generated_document'][0])
                sentences_answer.append(doc['generated_answer'][0])
                queries_ids.append(doc['query_id'])

            embeddings_doc = get_query_embeddings_by_model(self.encoder_models,model, sentences_doc ,128,True)
            embeddings_answer = get_query_embeddings_by_model(self.encoder_models,model, sentences_answer ,128,True)
            
            results_doc = get_inference_results(embeddings_doc, queries_ids, collection_name, model, top_k)
            results_answer = get_inference_results(embeddings_answer, queries_ids, collection_name, model, top_k)
            results = self.rrf_doc_answer_results(queries_ids,results_doc, results_answer)             
            with open(os.path.join(path, f"results_{model}.json"), "w") as f:
                    json.dump(results, f, indent=2)
                
    def get_o_query_doc__results_rrf_from_jsonl_fire(self, path_doc_gen, path='../results/fire/doc5/',batch_size=32, top_k=10, top_k_final=10):
        os.makedirs(path, exist_ok=True)
        datos_por_vector = self.get_datos_fire_per_dataset(path_doc_gen)
        collection_name= f"kdir_{self.dataset_name}"
        for model in datos_por_vector:
            sentences_doc = []
            sentences_answer = []
            queries_ids = []
            for doc in datos_por_vector[model]:
                sentences_doc.append(doc['generated_document'][0])
                sentences_answer.append(doc['query'][0])
                queries_ids.append(doc['query_id'])

            embeddings_doc = get_query_embeddings_by_model(self.encoder_models,model, sentences_doc ,128,True)
            embeddings_answer = get_query_embeddings_by_model(self.encoder_models,model, sentences_answer ,128,True)
            
            results_doc = get_inference_results(embeddings_doc, queries_ids, collection_name, model, top_k)
            results_answer = get_inference_results(embeddings_answer, queries_ids, collection_name, model, top_k)
            results = self.rrf_doc_answer_results(queries_ids,results_doc, results_answer)             
            with open(os.path.join(path, f"results_{model}.json"), "w") as f:
                    json.dump(results, f, indent=2)

    def get_o_query_answer_results_rrf_from_jsonl_fire(self, path_doc_gen, path='../results/fire/doc5/',batch_size=32, top_k=10, top_k_final=10):
        os.makedirs(path, exist_ok=True)
        datos_por_vector = self.get_datos_fire_per_dataset(path_doc_gen)
        collection_name= f"kdir_{self.dataset_name}"
        for model in datos_por_vector:
            sentences_doc = []
            sentences_answer = []
            queries_ids = []
            for doc in datos_por_vector[model]:
                sentences_doc.append(doc['query'][0])
                sentences_answer.append(doc['generated_answer'][0])
                queries_ids.append(doc['query_id'])

            embeddings_doc = get_query_embeddings_by_model(self.encoder_models,model, sentences_doc ,128,True)
            embeddings_answer = get_query_embeddings_by_model(self.encoder_models,model, sentences_answer ,128,True)
            
            results_doc = get_inference_results(embeddings_doc, queries_ids, collection_name, model, top_k)
            results_answer = get_inference_results(embeddings_answer, queries_ids, collection_name, model, top_k)
            results = self.rrf_doc_answer_results(queries_ids,results_doc, results_answer)             
            with open(os.path.join(path, f"results_{model}.json"), "w") as f:
                    json.dump(results, f, indent=2)


    def get_results_only_one_from_jsonl_fire(self, pseudo_doc ,path_doc_gen, path='../results/fire/doc5/',batch_size=32, top_k=10, top_k_final=10):
        os.makedirs(path, exist_ok=True)
        datos_por_vector = self.get_datos_fire_per_dataset(path_doc_gen)
        collection_name= f"kdir_{self.dataset_name}"
        for model in datos_por_vector:
            sentences_doc = []
            queries_ids = []
            for doc in datos_por_vector[model]:
                sentences_doc.append(doc[pseudo_doc][0])
                queries_ids.append(doc['query_id'])
            embeddings_doc = get_query_embeddings_by_model(self.encoder_models,model, sentences_doc ,128,True)
            results_doc = get_inference_results(embeddings_doc, queries_ids, collection_name, model, top_k)
            with open(os.path.join(path, f"results_{model}.json"), "w") as f:
                    json.dump(results_doc, f, indent=2)


    def count_bge_tokens_input(self):
        sentences, queries_ids, all_rel_docs = self.get_all_rel_docs(32, 5)
        documents_input=[]
        queries_input=[]
        anwers_input=[]
        for i, query in tqdm(enumerate(sentences), desc='Processing queries (generate or skip)'):
            rel_docs = all_rel_docs[3]
            prompts = self.promptor.get_prompt(query, rel_docs[i])
            doc = prompts[0]
            documents_input.append(doc)
            input_query = prompts[1]
            queries_input.append(input_query)
            answer = prompts[2]
            anwers_input.append(answer)
        return {'doc_input':documents_input, 'query_input':queries_input, 'anwers_input':anwers_input}

    


    def generate_openai_doc_and_save_prf(self, gen_model, path_input, path_to_save='../doc_gen/fire/openai/'):
        model ="bge_large"
        sentences, queries_ids = get_sentences_queries_openai(self.dataset_name)
        with open(path_input, 'r', encoding='utf-8') as archivo:
            datos = json.load(archivo)
        os.makedirs(path_to_save+f'{model}/', exist_ok=True)
        output_filepath = os.path.join(path_to_save+f'{model}/', f"{gen_model}_generated_documents_{self.dataset_name}.jsonl")
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
            for i, prompt in tqdm(enumerate(datos['doc_input']), desc='Processing queries (generate or skip)'):
                current_query_id = queries_ids[i]
                if current_query_id in existing_query_ids:
                    continue
                #print("=======")
                #print(prompt)
                ans1 = self.generate(prompt)
                entry = {
                    "query_id": current_query_id,
                    "generated_document": ans1,
                }
                f_write.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"Document generation complete. Results saved in: {output_filepath}")

    def generate_and_save_prf_bge_large(self, path='../doc_gen/fire/', batch_size=32, top_k=5, max_retries=5, retry_delay_seconds=1, rate_limit_delay_seconds=60):
        vectors =["contriever","contriever_ft","dpr","bge_large","gte_large","sparse_bm25"]
        sentences, queries_ids, all_rel_docs = self.get_all_rel_docs(batch_size, top_k)
        for j, rel_docs in enumerate(all_rel_docs):
            os.makedirs(path+f'{vectors[j]}/', exist_ok=True)
            output_filepath = os.path.join(path+f'{vectors[j]}/', f"generated_documents_{self.dataset_name}.jsonl")
            if(vectors[j]=="bge_large"):
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
                        prompts = self.promptor.get_prompt(query, rel_docs[i])
                        ans1 = self.generate(prompts[0], max_retries=max_retries, retry_delay_seconds=retry_delay_seconds, rate_limit_delay_seconds=rate_limit_delay_seconds)
                        ans2 = self.generate(prompts[1], max_retries=max_retries, retry_delay_seconds=retry_delay_seconds, rate_limit_delay_seconds=rate_limit_delay_seconds)
                        ans3 = self.generate(prompts[2], max_retries=max_retries, retry_delay_seconds=retry_delay_seconds, rate_limit_delay_seconds=rate_limit_delay_seconds)
                        entry = {
                            "query_id": current_query_id,
                            "query": query,
                            "generated_document": ans1,
                            "generated_query": ans2,
                            "generated_answer": ans3
                        }
                        f_write.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"Document generation complete. Results saved in: {output_filepath}")

    def get_datos_fire_per_model(self, path_doc_gen, model ):
        datos_por_vector = {}
        datos = []
        with open(path_doc_gen+f'{model}/generated_documents_{self.dataset_name}.jsonl', 'r', encoding='utf-8') as f:
            for linea in f:
                datos.append(json.loads(linea.strip()))
            datos_por_vector[model]=datos
        return datos_por_vector
    
    def get_results_by_model_fire(self, pseudo_doc ,path_doc_gen, path,batch_size=32, top_k=10, top_k_final=10):
        os.makedirs(path, exist_ok=True)
        datos_por_vector = self.get_datos_fire_per_model(path_doc_gen, 'bge_large')
        collection_name= f"kdir_{self.dataset_name}"
        for model in datos_por_vector:
            sentences_doc = []
            queries_ids = []
            for doc in datos_por_vector[model]:
                sentences_doc.append(doc[pseudo_doc][0])
                queries_ids.append(doc['query_id'])
            embeddings_doc = get_query_embeddings_by_model(self.encoder_models,model, sentences_doc ,128,True)
            results_doc = get_inference_results(embeddings_doc, queries_ids, collection_name, model, top_k)
            with open(os.path.join(path, f"results_{model}.json"), "w") as f:
                    json.dump(results_doc, f, indent=2)
    
    def get_3best_results_rrf_from_jsonl_fire_v2(self, path_doc_gen, path='../results/fire/doc5/',batch_size=32, top_k=10, top_k_final=10):
        os.makedirs(path, exist_ok=True)
        datos_por_vector = self.get_datos_fire_per_model(path_doc_gen, 'bge_large')
        collection_name= f"kdir_{self.dataset_name}"
        for model in datos_por_vector:
            query_list=[]
            sentences_doc = []
            sentences_answer = []
            queries_ids = []
            for doc in datos_por_vector[model]:
                query_list.append(doc['query'])
                sentences_doc.append(doc['generated_document'][0])
                sentences_answer.append(doc['generated_answer'][0])
                queries_ids.append(doc['query_id'])

            embeddings_query_list = get_query_embeddings_by_model(self.encoder_models,model, query_list ,128,True)
            embeddings_doc = get_query_embeddings_by_model(self.encoder_models,model, sentences_doc ,128,True)
            embeddings_answer = get_query_embeddings_by_model(self.encoder_models,model, sentences_answer ,128,True)
            
            results_query = get_inference_results(embeddings_query_list, queries_ids, collection_name, model, top_k)
            results_doc = get_inference_results(embeddings_doc, queries_ids, collection_name, model, top_k)
            results_answer = get_inference_results(embeddings_answer, queries_ids, collection_name, model, top_k)
            results = self.rrf_doc_answer_query_results(queries_ids, results_query,results_doc, results_answer)             
            with open(os.path.join(path, f"results_{model}.json"), "w") as f:
                    json.dump(results, f, indent=2)


    



        