import os
import pandas as pd
from langchain.vectorstores import FAISS  # 사용 중인 FAISS vector DB 클래스
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from typing import List
import torch
from langchain_community.vectorstores import FAISS
import pandas as pd

# 1. HuggingFace Granite 임베딩 모델을 사용하는 커스텀 임베딩 클래스
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
# CSV 파일 경로 설정 (정제된 CSV 파일 경로)
csv_path = "backend/faiss_index/metadata.csv"  # 실제 파일 경로로 변경하세요

# CSV 파일 읽기
df = pd.read_csv(csv_path)

# Document 객체 리스트 생성
# 1차 매칭: llm_keywords를 임베딩 텍스트로 사용
# 2차 매칭: page_content (및 document_title)는 메타데이터에 저장
documents = []
for _, row in df.iterrows():
    document_title = row["document_title"]
    llm_keywords = row["llm_keywords"]
    page_content = row["page_content"]

    doc = Document(
        page_content=page_content,  # primary text for embedding
        metadata={
            "document_title": document_title,
            "llm_keywords": llm_keywords,
            "page_content": page_content
        }
    )
    documents.append(doc)

# 임베딩 객체 생성 (실제 사용 중인 임베딩 클래스 사용)
embeddings = GraniteEmbeddings()

# FAISS 벡터 DB 구축: 문서 객체와 임베딩을 사용하여 DB 생성
faiss_db = FAISS.from_documents(documents, embeddings)

# FAISS DB 저장: API에서 load_local("faiss_index", ...)로 호출할 수 있도록 동일 폴더 이름 사용
faiss_save_dir = "backend/faiss_index"
faiss_db.save_local(faiss_save_dir)

print("✅ FAISS DB가 성공적으로 구축되어 저장되었습니다!")

