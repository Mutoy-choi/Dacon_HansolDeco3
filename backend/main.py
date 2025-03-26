import os
import torch
import numpy as np
import pandas as pd
from typing import List
import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

from fastapi import FastAPI
from pydantic import BaseModel, Field
import ollama
import json
import math
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
import logging
from fastapi.responses import JSONResponse

# --- 추가: FAISS와 LLM 임베딩 관련 라이브러리 ---
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

# ✅ 로그 설정
LOG_FILE = "logs.txt"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")

# 커스텀 임베딩 클래스 (HuggingFace Granite 모델 사용)
class GraniteEmbeddings(Embeddings):
    def __init__(self, model_name: str = "ibm-granite/granite-embedding-107m-multilingual", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]
    
    def _embed(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 마지막 hidden state의 평균값을 임베딩 벡터로 사용
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embedding

def log_message(message: str):
    """로그 메시지를 파일에 저장"""
    logging.info(message)

app = FastAPI()

# ✅ CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 테스트 케이스 파일 로드 (test.csv)
TEST_FILE = "test.csv"
test_data = pd.read_csv(TEST_FILE)

# 전역 변수로 현재 테스트 케이스의 인덱스를 관리 (초기값 0)
current_test_index = 0

@app.get("/next_test_case")
def next_test_case():
    global current_test_index
    if current_test_index >= len(test_data):
        return JSONResponse(content={"message": "모든 테스트 케이스가 처리되었습니다."})
    
    # test.csv 파일의 현재 행 가져오기 (예: 컬럼 이름 "사고객체", "사고원인")
    row = test_data.iloc[current_test_index]
    current_test_index += 1

    return {
        "accident_object": row["사고객체"],
        "accident_cause": row["사고원인"],
        "gongjong": row["공종"],
        "jobProcess": row["작업프로세스"],
        "location": row["장소"],
        "part": row["부위"],
        "humanAccident": row["인적사고"],
        "materialAccident": row["물적사고"],

    }

# ✅ API: 로그 가져오기
@app.get("/logs")
def get_logs():
    """저장된 로그를 클라이언트로 반환"""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = f.readlines()
        return JSONResponse(content={"logs": logs[-10:]})  # 최근 10개 로그 반환
    except FileNotFoundError:
        return JSONResponse(content={"logs": ["로그 파일이 없습니다."]})
    
# ✅ NumPy 타입을 Python 기본 타입으로 변환하는 함수
def convert_to_serializable(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(np.nan_to_num(obj))  # nan을 0.0으로 변환
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ✅ NaN 값 재귀적 변환 함수
def sanitize_data(data):
    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    elif isinstance(data, float):
        return 0.0 if math.isnan(data) else data
    else:
        return data

# ✅ 데이터 경로
DATA_FILE = "responses.jsonl"
VECTOR_FILE = "vectorized_data.npz"
DB_FILE = "database.csv"

# ✅ 데이터 로드 (학습 데이터)
train_data = pd.read_csv(DB_FILE)

# ✅ 최적 가중치 설정
optimal_weights = {
    '사고원인': 0.45,
    '공종': 0.15,
    '작업프로세스': 0.1,
    '인적사고': 0.0,
    '물적사고': 0.0,
    '장소': 0.05,
    '사고객체': 0.15,
    '부위': 0.1
}
factor_cols = list(optimal_weights.keys())

# ✅ 벡터 데이터 로드 및 정규화 (각 요소별 벡터)
vector_data = np.load(VECTOR_FILE)
vector_dict = {}
for factor in factor_cols:
    if factor == "사고원인":
        vector_dict[factor] = vector_data["사고원인"]
    elif factor == "사고객체":
        vector_dict[factor] = vector_data["사고객체"]
    else:
        vector_dict[factor] = vector_data[f"{factor}"]
    # L2 정규화 (0으로 나누는 것을 방지)
    vector_norm = np.linalg.norm(vector_dict[factor], axis=1, keepdims=True)
    vector_norm[vector_norm == 0] = 1
    vector_dict[factor] = vector_dict[factor] / vector_norm

# ✅ 모델 로드 (사고 요소 벡터화에는 SentenceTransformer 사용)
device = "cuda" if torch.cuda.is_available() else "cpu"
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("ibm-granite/granite-embedding-107m-multilingual", device=device)

# 기존 유사도 계산 함수 (레거시용: 사고객체와 사고원인만 사용)
def compute_similarity(vector, matrix):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros(matrix.shape[0])
    vector = vector / norm
    similarity = np.dot(matrix, vector.T)
    return np.nan_to_num(similarity)

def find_similar_accidents(object_text, cause_text, top_n=5):
    log_message(f"🔍 유사 사고 검색 시작: {object_text}, {cause_text}")

    test_object_vector = model.encode([object_text])[0]
    test_cause_vector = model.encode([cause_text])[0]

    object_norm = np.linalg.norm(test_object_vector)
    cause_norm = np.linalg.norm(test_cause_vector)

    if object_norm == 0:
        test_object_vector = np.zeros_like(test_object_vector)
    else:
        test_object_vector /= object_norm

    if cause_norm == 0:
        test_cause_vector = np.zeros_like(test_cause_vector)
    else:
        test_cause_vector /= cause_norm

    object_similarity_test = compute_similarity(test_object_vector, vector_dict["사고객체"])
    cause_similarity_test = compute_similarity(test_cause_vector, vector_dict["사고원인"])

    final_similarity_test = (object_similarity_test * optimal_weights["사고객체"]) + (cause_similarity_test * optimal_weights["사고원인"])
    final_similarity_test = np.nan_to_num(final_similarity_test)

    top_indices = np.argsort(-final_similarity_test)[:top_n]
    similar_cases = [
        {
            "case": sanitize_data(train_data.iloc[i].to_dict()),
            "similarity": convert_to_serializable(final_similarity_test[i]),
        }
        for i in top_indices
    ]

    log_message(f"✅ 유사 사고 검색 완료: {similar_cases}")

    return similar_cases

# --- 3. 테스트 데이터와 학습 데이터 간 유사도 계산 ---
# 테스트 데이터 벡터화 (각 요소별)
test_df = test_data  # test.csv 파일에 factor_cols에 해당하는 컬럼이 있다고 가정
test_vector_dict = {
    col: torch.tensor(
        model.encode(test_df[col].tolist(), convert_to_numpy=True),
        device=device,
        dtype=torch.float32
    )
    for col in factor_cols
}

similarity_matrices_test = {}
for col in factor_cols:
    # cosine_similarity는 numpy 배열을 요구하므로 GPU 텐서를 CPU로 이동 후 변환
    similarity_matrices_test[col] = cosine_similarity(
        test_vector_dict[col].cpu().numpy(),
        vector_dict[col]
    )

# --- 4. 최적 가중치를 반영하여 최종 유사도 행렬 생성 ---
final_similarity_test = np.zeros_like(list(similarity_matrices_test.values())[0])
for col in factor_cols:
    final_similarity_test += optimal_weights[col] * similarity_matrices_test[col]

# --- 5. 유사 사고 찾기 함수 (테스트 데이터 기준) ---
def find_similar_accidents_for_test(test_index, top_n=5):
    """
    주어진 테스트 샘플(test_index)의 여러 요소를 고려하여,
    학습 데이터(train_data)에서 가장 유사한 사례 top_n개를 반환합니다.
    """
    similarities = final_similarity_test[test_index]
    top_n = min(top_n, len(similarities))
    similar_indices = np.argsort(-similarities)[:top_n]
    return similar_indices, similarities[similar_indices]

# --- 중국어 감지 최적화 함수 ---
from langdetect import detect
def is_chinese(text):
    try:
        return detect(text) == "zh-cn"  # 중국어 감지
    except:
        return False  # 감지 실패 시 안전하게 처리

# --- 업데이트된 Ollama 요청 함수 및 프롬프트 ---
def generate_rag_response(top_k_cases, 기준_사고객체, 기준_사고원인, 기준_공종, 기준_작업프로세스, 기준_장소, 기준_부위, 기준_인적사고, 기준_물적사고):
    system_message = (
        "<|im_start|>system\n"
        "당신은 한국인 건설 사고 전문가입니다.\n"
        "주어진 유사 사고 사례들을 참고하여, 가장 효과적인 재발방지대책 및 향후 조치 계획을 작성해야 합니다.\n"
        "사고객체와 사고원인이 가장 중요한 요소이며, 인적사고와 물적사고도 중요한 고려 요소입니다.\n"
        "공종, 작업프로세스, 장소, 부위도 반영하여 최적의 대응책을 도출하세요.\n"
        "답변은 반드시 한국어로 작성해야 하며, 한 문장 내에서 각 요소를 반영하여 50단어 이내로 요약하세요.\n"
        "출력 형식: '대응대책:'으로 시작하는 한 문장으로 작성하세요.\n"
        "<|im_end|>"
    )

    user_message = (
        "<|im_start|>user\n"
        "[기준 사고]\n"
        f"- 사고객체: {기준_사고객체}\n"
        f"- 사고원인: {기준_사고원인}\n"
        f"- 인적사고: {기준_인적사고}\n"
        f"- 물적사고: {기준_물적사고}\n"
        f"- 공종: {기준_공종}\n"
        f"- 작업프로세스: {기준_작업프로세스}\n"
        f"- 장소: {기준_장소}\n"
        f"- 부위: {기준_부위}\n\n"
        "[유사 사례 - 기존 대응 대책들]:\n" +
        "\n".join([
            f"{i+1}. 사고객체: {case['사고객체']}, 사고원인: {case['사고원인']}, 인적사고: {case['인적사고']}, 물적사고: {case['물적사고']}, 공종: {case['공종']}, "
            f"작업프로세스: {case['작업프로세스']}, 장소: {case['장소']}, 부위: {case['부위']}, 대응 대책: {case['재발방지대책 및 향후조치계획']}"
            for i, case in enumerate(top_k_cases)
        ]) + "\n\n"
        "위의 유사 사례들을 참고하여, 기준 사고에 대한 최적의 재발방지대책 및 향후 조치 계획을 작성하세요.\n"
        "유사 사례가 부족한 경우, 기존 대응 대책을 일반화하여 최적의 해결책을 도출하세요.\n"
        "출력 형식: 반드시 '대응대책:'으로 시작하는 한 문장으로 작성하세요.\n"
        "<|im_end|>"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    while True:
        response = ollama.chat(model="qwq", messages=messages)
        output = response["message"]["content"]

        return output


# ✅ API 요청 모델 (업데이트된 사고 입력)
class QueryRequest(BaseModel):
    사고객체: str
    사고원인: str
    공종: str
    작업프로세스: str
    장소: str
    부위: str
    인적사고: str
    물적사고: str


# ✅ API: 사고 입력을 받아 LLM 응답 생성 (업데이트됨)
@app.post("/generate_responses")
def generate_responses(request: QueryRequest):
    # 입력 받은 데이터 로그 기록
    log_message(f"LLM 요청 수신: {request.dict()}")
    
    # 각 요소별 query 벡터 생성 (정규화 포함)
    query_vectors = {}
    for factor in optimal_weights.keys():
        text = getattr(request, factor)
        vec = model.encode(text)
        norm = np.linalg.norm(vec)
        if norm != 0:
            vec /= norm
        query_vectors[factor] = vec

    # 각 요소별 유사도 계산 후 가중치 반영하여 최종 유사도 산출
    similarities = np.zeros(train_data.shape[0])
    for factor in optimal_weights.keys():
        sim = np.dot(vector_dict[factor], query_vectors[factor].T)
        similarities += optimal_weights[factor] * sim

    top_n = 5
    top_indices = np.argsort(-similarities)[:top_n]
    
    # 각 케이스를 딕셔너리 형태로 구성 (유사도도 포함)
    top_k_cases = [
        {
            "사고객체": train_data.iloc[i]["사고객체"],
            "사고원인": train_data.iloc[i]["사고원인"],
            "인적사고": train_data.iloc[i]["인적사고"],
            "물적사고": train_data.iloc[i]["물적사고"],
            "공종": train_data.iloc[i]["공종"],
            "작업프로세스": train_data.iloc[i]["작업프로세스"],
            "장소": train_data.iloc[i]["장소"],
            "부위": train_data.iloc[i]["부위"],
            "재발방지대책 및 향후조치계획": train_data.iloc[i]["재발방지대책 및 향후조치계획"],
            "similarity": float(similarities[i]),
        }
        for i in top_indices
    ]
    log_message(f"유사 사고 케이스 도출: {top_k_cases}")

    response = generate_rag_response(
        top_k_cases,
        기준_사고객체=request.사고객체,
        기준_사고원인=request.사고원인,
        기준_공종=request.공종,
        기준_작업프로세스=request.작업프로세스,
        기준_장소=request.장소,
        기준_부위=request.부위,
        기준_인적사고=request.인적사고,
        기준_물적사고=request.물적사고
    )
    
    log_message(f"LLM 응답 생성 완료: {response}")
    
    return {
        "query": {
            "사고객체": request.사고객체,
            "사고원인": request.사고원인,
            "공종": request.공종,
            "작업프로세스": request.작업프로세스,
            "장소": request.장소,
            "부위": request.부위,
            "인적사고": request.인적사고,
            "물적사고": request.물적사고
        },
        "top_cases": top_k_cases,
        "answer": response,
    }


# ✅ API 요청 모델 (피드백 저장)
class FeedbackRequest(BaseModel):
    query: str
    winner: str
    loser: str

# ✅ API: 피드백 저장
@app.post("/save_feedback")
def save_feedback(request: FeedbackRequest):
    feedback_data = {"query": request.query, "winner": request.winner, "loser": request.loser}
    with open(DATA_FILE, "a", encoding="utf-8") as f:
        json.dump(feedback_data, f, ensure_ascii=False)
        f.write("\n")
    return {"message": "Feedback saved successfully!"}

# --- 추가: FAISS 기반 문서 검색 API ---
@app.get("/get_documents")
def get_documents(accident_cause: str = None):
    if not accident_cause:
        accident_cause = str(test_data.iloc[0]["사고원인"]).strip()
    log_message(f"문서 검색 요청: 사고원인={accident_cause}")
    
    try:
        embeddings = GraniteEmbeddings()
        vectordb = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        log_message(f"FAISS 인덱스 로드 실패: {e}")
        return JSONResponse(content={"error": f"FAISS 인덱스 로드 실패: {e}"})
    
    results_with_scores = vectordb.similarity_search_with_score(accident_cause, k=5)
    min_llm_keywords_similarity_threshold = 0.65
    accident_cause_emb = embeddings.embed_query(accident_cause)
    
    final_results = []
    for doc, score in results_with_scores:
        llm_keywords = doc.metadata.get("llm_keywords", "")
        if llm_keywords:
            llm_keywords_emb = embeddings.embed_query(llm_keywords)
            sim = np.dot(np.array(accident_cause_emb), np.array(llm_keywords_emb)) / (
                (np.linalg.norm(accident_cause_emb) * np.linalg.norm(llm_keywords_emb)) + 1e-8)
            if sim >= min_llm_keywords_similarity_threshold:
                result_dict = {
                    "title": doc.metadata.get("document_title", "No Title"),
                    "faiss_score": float(score),
                    "llm_keywords_similarity": float(sim),
                    "llm_keywords": llm_keywords,
                    "metadata": doc.metadata,
                    "chunk_content": doc.page_content
                }
                final_results.append(result_dict)
    if not final_results:
        log_message("문서 검색: 유사도 임계치를 만족하는 문서를 찾지 못함")
        return {"message": "유사도 임계치를 만족하는 문서를 찾지 못했습니다."}
    
    log_message(f"문서 검색 완료: {len(final_results)}개 문서 반환")
    return {
        "accident_cause": accident_cause,
        "documents": final_results
    }

@app.get("/")
def read_root():
    return {"message": "LLM Feedback System is running!"}
