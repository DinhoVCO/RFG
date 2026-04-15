import numpy as np
from kdir_src.models.all_models import get_query_embeddings
from qdrant_client.models import SparseVector
from kdir_src.utils.qdrant import recuperar_documentos, recuperar_documentos_sparsos
from kdir_src.dataset.beir import get_sentences_queries
from kdir_src.utils.inference import get_inference_results
from tqdm import tqdm
import os
import json
import time

WEB_SEARCH = """Please write a passage to answer the question.
Question: {}
Passage:"""


SCIFACT = """Please write a scientific paper passage to support/refute the claim.
Claim: {}
Passage:"""


ARGUANA = """Please write a counter argument for the passage.
Passage: {}
Counter Argument:"""


NFCORPUS = """Please write a medical article passage to answer the question.
Question: {}
Passage:"""

TREC_COVID = """Please write a scientific paper passage to answer the question.
Question: {}
Passage:"""


FIQA = """Please write a financial article passage to answer the question.
Question: {}
Passage:"""


DBPEDIA_ENTITY = """Please write a passage to answer the question.
Question: {}
Passage:"""


TREC_NEWS = """Please write a news passage about the topic.
Topic: {}
Passage:"""


MR_TYDI = """Please write a passage in {} to answer the question in detail.
Question: {}
Passage:"""


SCIDOCS = """Please write a scientific document passage on the topic.
Topic: {}
Passage:"""

class Promptor:
    def __init__(self, task: str, language: str = 'en'):
        self.task = task
        self.language = language
    
    def build_prompt(self, query: str):
        if self.task == 'web search':
            return WEB_SEARCH.format(query)
        elif self.task == 'scifact':
            return SCIFACT.format(query)
        elif self.task == 'arguana':
            return ARGUANA.format(query)
        elif self.task == 'nfcorpus':
            return NFCORPUS.format(query)
        elif self.task == 'trec-covid':
            return TREC_COVID.format(query)
        elif self.task == 'fiqa':
            return FIQA.format(query)
        elif self.task == 'dbpedia-entity':
            return DBPEDIA_ENTITY.format(query)
        elif self.task == 'trec-news':
            return TREC_NEWS.format(query)
        elif self.task == 'mr-tydi':
            return MR_TYDI.format(self.language, query)
        elif self.task == 'scidocs':
            return SCIDOCS.format(query)
        else:
            raise ValueError('Task not supported')

def average_sparse_vectors(list_of_sparse_vectors):
    """
    Calcula una "incrustación promedio" sumando los pesos de términos comunes
    a través de múltiples SparseVectors.

    Args:
        list_of_sparse_vectors (list): Una lista de objetos models.SparseVector.

    Returns:
        models.SparseVector: Un nuevo SparseVector que representa la incrustación promedio.
    """
    combined_weights = {} # Diccionario para acumular la suma de pesos por índice

    for sv in list_of_sparse_vectors:
        # Acceder a los atributos indices y values del objeto SparseVector
        for i, val in zip(sv.indices, sv.values):
            combined_weights[i] = combined_weights.get(i, 0.0) + val

    # Crear el nuevo SparseVector a partir de los pesos combinados
    # Ordenar los índices es una buena práctica para consistencia
    averaged_indices = sorted(combined_weights.keys())
    averaged_values = [combined_weights[idx] for idx in averaged_indices]

    return SparseVector(indices=averaged_indices, values=averaged_values)


class HyDE:
    def __init__(self, datset_name, promptor, generator, encoder_models):
        self.promptor = promptor
        self.generator = generator
        self.encoder_models = encoder_models
        self.dataset_name = datset_name
    
    def prompt(self, query):
        return self.promptor.build_prompt(query)

    def generate(self, query, max_retries=5, retry_delay_seconds=1):
        """
        Generates hypothetical documents for a given query, with retry logic for API errors.

        Args:
            query (str): The input query.
            max_retries (int): Maximum number of retries if the API call fails.
            retry_delay_seconds (int): Delay in seconds between retries.

        Returns:
            list: A list of generated documents (strings).
            
        Raises:
            Exception: If generation fails after all retries.
        """
        prompt = self.promptor.build_prompt(query)
        
        for attempt in range(max_retries):
            try:
                # Llama a tu modelo generativo (LLM real) aquí
                # Asegúrate de que self.generator.generate(prompt) devuelva una lista de strings.
                hypothesis_documents = self.generator.generate(prompt) 
                
                # print(f"  --> Attempt {attempt + 1}: Generated for query: '{query}'") # Opcional para depuración
                return hypothesis_documents
            except Exception as e: # Captura excepciones generales. Puedes ser más específico si conoces los errores de tu API.
                print(f"Error generating for query '{query}' (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay_seconds} seconds...")
                    time.sleep(retry_delay_seconds)
                else:
                    print(f"Max retries reached for query '{query}'. Giving up.")
                    raise # Vuelve a lanzar la excepción si se agotaron los reintentos


    # def generate(self, query):
    #     prompt = self.promptor.build_prompt(query)
    #     hypothesis_documents = self.generator.generate(prompt)
    #     return hypothesis_documents
    
    def encode(self, query, hypothesis_documents):
        avg_vector_models=[]
        c_emb_list = get_query_embeddings(self.encoder_models, [query] + hypothesis_documents,32,False)
        for embeding_by_model in c_emb_list[:-1]:
            all_emb_c = np.array(embeding_by_model)
            avg_emb_c = np.mean(all_emb_c, axis=0)
            hyde_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
            avg_vector_models.append(hyde_vector[0])
        sparse_bm25 = average_sparse_vectors(c_emb_list[-1])
        avg_vector_models.append(sparse_bm25)
        return avg_vector_models

    def search_by_query(self, query, collection_name, top_k=10):
        hypothesis_documents = self.generate(query)
        hyde_vector = self.encode(query, hypothesis_documents)
        vectors =["contriever","contriever_ft","dpr","bge_large","gte_large","sparse_bm25"]
        all_models_rel_docs=[]
        for i, embedding in enumerate(hyde_vector[:-1]):
            if(vectors[i]!="sparse_bm25"):
                rel_docs = recuperar_documentos(collection_name, embedding[0], vectors[i], top_k)
                all_models_rel_docs.append(rel_docs)
            else:
                rel_docs = recuperar_documentos_sparsos(collection_name, embedding, vectors[i], top_k)
                all_models_rel_docs.append(rel_docs)
        return all_models_rel_docs



    def save_generated_docs(self, path='../doc_gen/hyde/'):
        """
        Generates hypothetical documents for each query in the dataset and saves them
        to a JSONL (JSON Lines) file incrementally, skipping already generated ones.

        Args:
            path (str): The directory path where the JSONL file will be saved.
        """
        os.makedirs(path, exist_ok=True) # Ensure the directory exists
        sentences, queries_ids = get_sentences_queries(self.dataset_name)
        output_filepath = os.path.join(path, f"generated_documents_{self.dataset_name}.jsonl")
        
        # --- Step 1: Read existing generated query IDs ---
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

        # --- Step 2: Open file in append mode and iterate through queries ---
        # Open in 'a' (append) mode so new lines are added without overwriting
        # If the file didn't exist, 'a' will create it.
        with open(output_filepath, 'a', encoding='utf-8') as f_write:
            for i, query in tqdm(enumerate(sentences), desc='Processing queries (generate or skip)'):
                current_query_id = queries_ids[i]

                if current_query_id in existing_query_ids:
                    # If the query ID already exists, skip generation
                    # print(f"Skipping query ID '{current_query_id}' as it's already generated.")
                    continue
                
                # If not generated, proceed with generation
                hypothesis_documents = self.generate(query)
                
                # Create the JSON object for the current line
                entry = {
                    "query_id": current_query_id,
                    "query": query,
                    "generated_documents": hypothesis_documents
                }
                
                # Convert the object to a JSON string and write the line, followed by a newline
                f_write.write(json.dumps(entry, ensure_ascii=False) + '\n')
                # The write is performed immediately after generating each set of documents.
        
        print(f"Document generation complete. Results saved in: {output_filepath}")



    def get_all_querys_embeddings(self):
        sentences, queries_ids = get_sentences_queries(self.dataset_name)
        contriever=[]
        contriever_ft=[]
        dpr=[]
        bge_large=[]
        gte_large=[]
        sparse_bm25=[]
        for query in tqdm(sentences, desc='getting embeddings of all queries'):
            hypothesis_documents = self.generate(query)
            hyde_vector = self.encode(query, hypothesis_documents)
            contriever.append(hyde_vector[0])
            contriever_ft.append(hyde_vector[1])
            dpr.append(hyde_vector[2])
            bge_large.append(hyde_vector[3])
            gte_large.append(hyde_vector[4])
            sparse_bm25.append(hyde_vector[5])
        all_querys_embeddings=[contriever,contriever_ft, dpr, bge_large, gte_large, sparse_bm25]
        return all_querys_embeddings, queries_ids

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
            hyde_vector = self.encode(doc['query'], hypothesis_documents)
            contriever.append(hyde_vector[0])
            contriever_ft.append(hyde_vector[1])
            dpr.append(hyde_vector[2])
            bge_large.append(hyde_vector[3])
            gte_large.append(hyde_vector[4])
            sparse_bm25.append(hyde_vector[5])
        all_querys_embeddings=[contriever,contriever_ft, dpr, bge_large, gte_large, sparse_bm25]
        return all_querys_embeddings, queries_ids

    def get_results_from_jsonl_hyde(self, path_doc_gen, path='../results/',batch_size=32, top_k=10):
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
            results = get_inference_results(model_embeddings, queries_ids, collection_name, vectors[i], top_k)
            with open(path+f"results_{vectors[i]}.json", "w") as f:
                json.dump(results, f, indent=2)
            i+=1

    def get_results_hyde(self, path='../results/',batch_size=32, top_k=10):
        os.makedirs(path, exist_ok=True)
        all_sentence_embeddings, queries_ids = self.get_all_querys_embeddings()
        collection_name= f"kdir_{self.dataset_name}"
        vectors =["contriever","contriever_ft","dpr","bge_large","gte_large","sparse_bm25"]
        i=0
        for model_embeddings in all_sentence_embeddings:
            results = get_inference_results(model_embeddings, queries_ids, collection_name, vectors[i], top_k)
            with open(path+f"results_{vectors[i]}.json", "w") as f:
                json.dump(results, f, indent=2)
            i+=1
        
        
        
            
        
        