import os
import torch
import numpy as np
import pandas as pd
from typing import List
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from langchain.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

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

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

if __name__ == "__main__":
    # 저장된 FAISS 인덱스 로드 (allow_dangerous_deserialization 옵션 사용)
    index_dir = "backend/faiss_index"
    
    # FAISS DB 로드 시 올바른 언패킹 적용
    try:
        vectordb = FAISS.load_local(index_dir, embeddings=GraniteEmbeddings(), allow_dangerous_deserialization=True)
        print("FAISS index loaded successfully.")
    except TypeError as e:
        print(f"FAISS 로드 오류: {e}")
        
        # 강제 언패킹 시도 (v0.1.x 이후 버전의 변경점 대응)
        faiss_index, metadata_store = FAISS.load_local(index_dir, GraniteEmbeddings(), allow_dangerous_deserialization=True)

        # `faiss_index`와 `metadata_store`가 올바르게 로드되었는지 확인
        if isinstance(metadata_store, InMemoryDocstore):
            print("📌 FAISS와 메타데이터가 정상적으로 로드되었습니다.")
            vectordb = FAISS(index=faiss_index, docstore=metadata_store, embedding_function=GraniteEmbeddings())
        else:
            raise ValueError("❌ 올바른 FAISS 메타데이터 형식이 아닙니다.")

    # test.csv 파일 로드 (사고 데이터가 CSV 형식으로 저장되어 있다고 가정)
    test_df = pd.read_csv("backend/test.csv")
    
    # 예시로 첫 번째 행의 데이터를 사용하여 사고원인 쿼리 문자열 생성 (사고원인 컬럼 사용)
    accident_cause = str(test_df.iloc[0]["사고원인"]).strip()
    print("Accident Cause Query:", accident_cause)
    
    # 1단계: FAISS DB에서 사고원인과 유사한 청크 검색 (유사도 점수와 함께)
    results_with_scores = vectordb.similarity_search_with_score(accident_cause, k=5)
    
    # 2단계: 각 청크의 metadata에 저장된 llm_keywords와 사고원인 간의 임베딩 유사도 계산
    min_llm_keywords_similarity_threshold = 0.7
    embedding_model = GraniteEmbeddings()
    accident_cause_emb = embedding_model.embed_query(accident_cause)
    
    final_results = []
    for doc, score in results_with_scores:
        llm_keywords = doc.metadata.get("llm_keywords", "")
        if llm_keywords:
            llm_keywords_emb = embedding_model.embed_query(llm_keywords)
            sim = cosine_similarity(accident_cause_emb, llm_keywords_emb)
            if sim >= min_llm_keywords_similarity_threshold:
                final_results.append((doc, score, sim))
        else:
            continue
    
    if not final_results:
        print("사고원인과 llm_keywords의 유사도가 임계치 이상인 결과가 없습니다.")
    else:
        print("\nAccident Cause Query:", accident_cause)
        print("Filtered Documents:")
        for i, (doc, score, llm_sim) in enumerate(final_results):
            title = doc.metadata.get("document_title", "No Title")
            llm_keywords = doc.metadata.get("llm_keywords", "N/A")
            metadata = doc.metadata
            chunk_content = doc.page_content
            print(f"\nDocument {i+1}:")
            print("Title:", title)
            print("FAISS Score:", score)
            print("llm_keywords 유사도:", llm_sim)
            print("LLM Keywords:", llm_keywords)
            print("Metadata:", metadata)
            print("Chunk Content:")
            print(chunk_content)
            print("-" * 40)
