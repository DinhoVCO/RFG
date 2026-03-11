import json
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from kdir_src.dataset.beir import get_sentences_queries
from kdir_src.sota.query2doc import generar_prompt_few_shot
from kdir_src.dataset.beir import get_sentences_queries, build_beir_random_examples, get_beir_train_dataset
import os
from tqdm import tqdm
from kdir_src.sota.RAG_fusion import RAG_fusion, Promptor as Promptor_rag_fusion
from kdir_src.generators.mistral_ai import MistralGenerator
from kdir_src.models.all_models import load_models
from kdir_src.fire.Fire import Fire
from kdir_src.fire.promptor_fire import Promptor_fire
from kdir_src.sota.query2doc import Query2Doc
from kdir_src.sota.hyde import HyDE, Promptor as PromptorHyde



def count_output_tokens(ruta_archivo, doc_name='generated_document', model_name='mistral-small-2503'):
    try:
        tokenizer = MistralTokenizer.from_model(model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    print(f"Processing file: {ruta_archivo} with the tokenizer of {model_name}\n")
    docs = []
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        for i, linea in enumerate(f):
            try:
                dato = json.loads(linea.strip())
                documents = dato.get(doc_name, [])
                len_docs=0
                for j, doc in enumerate(documents):
                    req_doc = ChatCompletionRequest(model=model_name, messages=[UserMessage(content=doc)])
                    tokens_documento = len(tokenizer.encode_chat_completion(req_doc).tokens)
                    len_docs+=tokens_documento
                docs.append(len_docs)
            except json.JSONDecodeError:
                print(f"Warning: Line {i + 1} is not a valid JSON and will be omitted.")
            except Exception as e:
                print(f"Error processing line {i + 1}: {e}")

    print("\n--- Process completed ---")
    return {"gen_docs":docs}


def calculate_avg(results):
    promedio = sum(results['gen_docs']) / len(results['gen_docs'])
    return promedio

def calculate_input_avg(input_docs):
    promedio = sum(input_docs) / len(input_docs)
    return promedio


def count_output_tokens_hyde(path_doc_gen, doc_name):
    results={}
    for nombre_archivo in os.listdir(path_doc_gen):
        if nombre_archivo.endswith('.jsonl'):
            sin_prefijo = nombre_archivo.removeprefix('generated_documents_')
            dataset_name = sin_prefijo.removesuffix('.jsonl')
            ruta_completa = os.path.join(path_doc_gen, nombre_archivo)
            result_hyde = count_output_tokens(ruta_completa, doc_name)
            results[dataset_name]=result_hyde
    return results

def count_output_tokens_hyde_q2d_rag_fusion(path_doc_gen, doc_name):
    results={}
    for nombre_archivo in os.listdir(path_doc_gen):
        if nombre_archivo.endswith('.jsonl'):
            sin_prefijo = nombre_archivo.removeprefix('generated_documents_')
            dataset_name = sin_prefijo.removesuffix('.jsonl')
            ruta_completa = os.path.join(path_doc_gen, nombre_archivo)
            result_hyde = count_output_tokens(ruta_completa, doc_name)
            promedio = calculate_avg(result_hyde)
            results[dataset_name]=promedio
    return results


def count_output_tokens_fire(path_doc_gen):
    results={}
    for nombre_archivo in os.listdir(path_doc_gen):
        if nombre_archivo.endswith('.jsonl'):
            sin_prefijo = nombre_archivo.removeprefix('generated_documents_')
            dataset_name = sin_prefijo.removesuffix('.jsonl')
            ruta_completa = os.path.join(path_doc_gen, nombre_archivo)
            result_doc = count_output_tokens(ruta_completa, 'generated_document')
            result_gen_query = count_output_tokens(ruta_completa, 'generated_query')
            result_answer = count_output_tokens(ruta_completa, 'generated_answer')
            promedio_doc = calculate_avg(result_doc)
            promedio_gen_query = calculate_avg(result_gen_query)
            promedio_answer = calculate_avg(result_answer)
            individual_results={'doc':promedio_doc, 'gen_query':promedio_gen_query, 'ans': promedio_answer }
            results[dataset_name]=individual_results
    return results


def get_all_output_tokens(path_save):
    
    r_hyde=count_output_tokens_hyde_q2d_rag_fusion('../doc_gen/hyde/', 'generated_documents')
    r_q2d=count_output_tokens_hyde_q2d_rag_fusion('../doc_gen/query2doc/', 'generated_documents')
    r_rag_fusion=count_output_tokens_hyde_q2d_rag_fusion('../doc_gen/rag_fusion/', 'generated_documents')
    r_fire=count_output_tokens_fire('../doc_gen/fire/doc5/bge_large/')
    results={'hyde':r_hyde, 'q2d':r_q2d, 'rag_fusion':r_rag_fusion, 'fire':r_fire}
    os.makedirs(path_save, exist_ok=True)
    with open(path_save+'aoutput_tokens.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ Results successfully saved in: {path_save}")



def count_input_tokens_hyde_rag_fusion(promptor, dataset_name, model_name='mistral-small-2503'):
    try:
        tokenizer = MistralTokenizer.from_model(model_name)
    except Exception as e:
        print(f"Error al cargar el tokenizador: {e}")
        return
    sentences, queries_ids = get_sentences_queries(dataset_name)
    docs=[]
    for j, sentece in enumerate(sentences):
        prompt=promptor.build_prompt(sentece)
        req_doc = ChatCompletionRequest(model=model_name, messages=[UserMessage(content=prompt)])
        tokens_documento = len(tokenizer.encode_chat_completion(req_doc).tokens)
        docs.append(tokens_documento)
    return {"input_query":docs}


def count_input_tokens_q2d(dataset_name, model_name='mistral-small-2503'):
    try:
        tokenizer = MistralTokenizer.from_model(model_name)
    except Exception as e:
        print(f"Error al cargar el tokenizador: {e}")
        return
    docs=[]
    corpus_train, queries_train, qrels_train =get_beir_train_dataset(dataset_name)
    sentences, queries_ids = get_sentences_queries(dataset_name)
    for i, query in tqdm(enumerate(sentences), desc='Processing queries (generate or skip)'):
        examples = build_beir_random_examples(corpus_train, queries_train, qrels_train, 4)
        prompt_final = generar_prompt_few_shot(examples, query)
        req_doc = ChatCompletionRequest(model=model_name, messages=[UserMessage(content=prompt_final)])
        tokens_documento = len(tokenizer.encode_chat_completion(req_doc).tokens)
        docs.append(tokens_documento)
    return {"input_query":docs}


def count_input_token_bge_fire(couter_funcion, model_name='mistral-small-2503'):
    try:
        tokenizer = MistralTokenizer.from_model(model_name)
    except Exception as e:
        print(f"Error al cargar el tokenizador: {e}")
        return
    results = couter_funcion()
    docs=[]
    query=[]
    ans=[]
    for i in tqdm(range(len(results['doc_input'])), desc='Processing queries (generate or skip)'):
        req_doc = ChatCompletionRequest(model=model_name, messages=[UserMessage(content=results['doc_input'][i])])
        tokens_doc = len(tokenizer.encode_chat_completion(req_doc).tokens)
        docs.append(tokens_doc)
        req_query = ChatCompletionRequest(model=model_name, messages=[UserMessage(content=results['query_input'][i])])
        tokens_query = len(tokenizer.encode_chat_completion(req_query).tokens)
        query.append(tokens_query)
        req_ans = ChatCompletionRequest(model=model_name, messages=[UserMessage(content=results['doc_input'][i])])
        tokens_ans = len(tokenizer.encode_chat_completion(req_ans).tokens)
        ans.append(tokens_ans)
    return {'input_query_docs':docs, 'input_query_query':query, 'input_query_answer':ans}


def get_all_input_tokens(path_save):
    datasets=['arguana','scifact','scidocs','fiqa','nfcorpus']
    hyde={}
    q2d={}
    rag_fusion={}
    fire={}
    #results={'hyde'}
    encoder_models = load_models()
    for dataset_name in datasets:
        KEY = ''
        MODEL_MISTRAL='mistral-small-2503'
        # HYDE
        promptor_hyde = PromptorHyde(dataset_name)
        results_hyde = count_input_tokens_hyde_rag_fusion(promptor_hyde, dataset_name, MODEL_MISTRAL)
        avg_hyde = calculate_input_avg(results_hyde['input_query'])
        hyde[dataset_name] = avg_hyde
        if(dataset_name not in ['arguana', 'scidocs']):
            #Q2D
            results_q2d = count_input_tokens_q2d(dataset_name, MODEL_MISTRAL)
            avg_q2d = calculate_input_avg(results_q2d['input_query'])
            q2d[dataset_name]=avg_q2d
        #rag_fusion
        promptor_rag_fusion = Promptor_rag_fusion(dataset_name)
        results_rag_fusion= count_input_tokens_hyde_rag_fusion(promptor_rag_fusion, dataset_name, MODEL_MISTRAL)
        avg_rag_fusion = calculate_input_avg(results_rag_fusion['input_query'])
        rag_fusion[dataset_name] = avg_rag_fusion

        #FIRE
        generator = MistralGenerator('mistral-small-2503', KEY, 1, 512, 0.0)
        promptor = Promptor_fire(dataset_name)
        fire_rag = Fire(dataset_name, promptor, generator, encoder_models)
        results_bge_fire = count_input_token_bge_fire(fire_rag.count_bge_tokens_input, model_name='mistral-small-2503')
        avg_fire_doc = calculate_input_avg(results_bge_fire['input_query_docs'])
        avg_fire_query = calculate_input_avg(results_bge_fire['input_query_query'])
        avg_fire_answer = calculate_input_avg(results_bge_fire['input_query_answer'])
        fire[dataset_name]={'doc':avg_fire_doc, 'query':avg_fire_query, 'answer':avg_fire_answer}
    results ={'hyde':hyde, 'q2d':q2d, 'rag_fusion':rag_fusion, 'fire':fire}
    os.makedirs(path_save, exist_ok=True)
    with open(path_save+'input_tokens.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ Results successfully saved in: {path_save}")









def count_basic_input_tokens(dataset_name, model_name='mistral-small-2503'):
    try:
        tokenizer = MistralTokenizer.from_model(model_name)
    except Exception as e:
        print(f"Error al cargar el tokenizador: {e}")
        return
    docs=[]
    sentences, queries_ids = get_sentences_queries(dataset_name)
    for i, query in tqdm(enumerate(sentences), desc='Processing queries (generate or skip)'):
        req_doc = ChatCompletionRequest(model=model_name, messages=[UserMessage(content=query)])
        tokens_documento = len(tokenizer.encode_chat_completion(req_doc).tokens)
        docs.append(tokens_documento)
    return {"input_query":docs}

def get_all_basic_input_tokens():
    datasets=['arguana','scifact','scidocs','fiqa','nfcorpus']
    results={}
    for dataset in datasets:
        result= count_basic_input_tokens(dataset)
        avg = calculate_input_avg(result['input_query'])
        results[dataset]=avg
    return results


def get_input_prompt_fire_bge(dataset_name, path_save):
    KEY = ''
    MODEL_MISTRAL='mistral-small-2503'
    generator = MistralGenerator(MODEL_MISTRAL, KEY, 1, 512, 0.0)
    promptor = Promptor_fire(dataset_name)
    encoder_models = load_models()
    fire_rag = Fire(dataset_name, promptor, generator, encoder_models)
    results_bge_fire = fire_rag.count_bge_tokens_input()
    os.makedirs(path_save, exist_ok=True)
    with open(path_save+'input_prompt.json', 'w', encoding='utf-8') as f:
        json.dump(results_bge_fire, f, indent=4, ensure_ascii=False)

    print(f"✅ Results successfully saved in: {path_save}")
    