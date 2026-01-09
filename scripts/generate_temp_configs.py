
import json
import os

queries_raw = """latest dolls for girls age 7
baby dolls for girls
10 yr old presents for girls
Intelligent sensing floating ball
14”Kids bikes
14”wheelKids bikes
15inch girls bike
Age 3 to 6 with animals or dinosaurs
Age 6 to 8 hello neighbor trasformers
airoplane for toddlers
Boys trucks 69.99
Bracelet making set 
bumblee bee transformer
Captain America civil war action battle lego
chistmas for a 10 year old girl
Gift for 12 year old non verbal boy
Paint brush set
Skateboard for 12yrs with skull
Animatics
Tie down kit for 12x8 trampoline
Guigui"""

queries = [q.strip() for q in queries_raw.split('\n') if q.strip()]

models = [
    {
        "id": "minilm",
        "name": "all-MiniLM-L6-v2",
        "collection": "prd_minilm_l6v2_bm25_tfidf",
        "prefix": None
    },
    {
        "id": "e5_ml",
        "name": "intfloat/multilingual-e5-base",
        "collection": "prd_e5_ml_bm25_tfidf",
        "prefix": "query: "
    },
    {
        "id": "e5_small",
        "name": "intfloat/multilingual-e5-small",
        "collection": "prd_e5_small_bm25_tfidf",
        "prefix": "query: "
    },
    {
        "id": "bge_small",
        "name": "BAAI/bge-small-en-v1.5",
        "collection": "prd_bge_small_bm25_tfidf",
        "prefix": "Represent this sentence for searching relevant passages: "
    },
    {
        "id": "mxbai_large",
        "name": "mixedbread-ai/mxbai-embed-large-v1",
        "collection": "prd_mxbai_large_bm25_tfidf",
        "prefix": "Represent this sentence for searching relevant passages: "
    },
    {
        "id": "qwen3",
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "collection": "prd_qwen3_0.6b_bm25_tfidf",
        "prefix": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    }
]
temp_folder_name = "temp_20queries"
os.makedirs(f"configs/{temp_folder_name}", exist_ok=True)

for model in models:
    config = {
        "collection_name": model["collection"],
        "dense_model": model["name"],
        "sparse_model_path": "models/sparse_model.pkl",
        "queries": []
    }
    if model["prefix"]:
        config["dense_prefix"] = model["prefix"]
    
    for q_text in queries:
        # 1. Pure Dense
        config["queries"].append({
            "name": f"dense_{q_text}",
            "type": "dense",
            "query_text": q_text,
            "limit": 5
        })
        # 2. Pure Sparse
        config["queries"].append({
            "name": f"sparse_{q_text}",
            "type": "sparse",
            "query_text": q_text,
            "limit": 5
        })
        # 3. Hybrid
        config["queries"].append({
            "name": f"hybrid_{q_text}",
            "type": "hybrid",
            "query_text": q_text,
            "limit": 5
        })
        # 4. Threshold Dense
        config["queries"].append({
            "name": f"threshold_{q_text}",
            "type": "dense",
            "query_text": q_text,
            "score_threshold": 0.7,
            "limit": 5
        })
    
    file_path = f"configs/{temp_folder_name}/query_{model['id']}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"Generated {file_path}")
