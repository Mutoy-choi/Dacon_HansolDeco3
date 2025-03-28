import os
import torch
import numpy as np
from typing import List
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
# 최신 API에 맞는 SemanticChunker 사용
from langchain_experimental.text_splitter import SemanticChunker

# 최신 LangChain 경고에 따라 community 모듈 사용
from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain_community.vectorstores import FAISS

# 추가: Ollama LLM을 사용하기 위한 프롬프트 관련 라이브러리
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

# tqdm 진행 표시줄
from tqdm import tqdm
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

# 2. HuggingFaceEmbeddings를 사용하여 SemanticChunker에 넣을 임베딩 인스턴스 생성
embedding = HuggingFaceEmbeddings(model_name="ibm-granite/granite-embedding-107m-multilingual")

# SemanticChunker를 최신 인터페이스에 맞게 초기화
text_splitter = SemanticChunker(
    embeddings=embedding,
    buffer_size=1,
    add_start_index=False,
    breakpoint_threshold_type="percentile",  # 또는 'standard_deviation', 'interquartile', 'gradient'
    breakpoint_threshold_amount=0.7,         # 청크 병합 기준 임계값
    number_of_chunks=None,
    sentence_split_regex=r"(?<=[.?!])\s+",
    min_chunk_size=500  # 최소 청크 크기를 1000으로 설정 (이전의 chunk_size 대신 사용)
)

# 3. Ollama LLM 기반 키워드 추출 함수 (실시간으로 결과 출력)
def add_llm_keywords_to_documents(documents: List[Document]) -> List[Document]:
    """
    각 청크에 대해 다음 항목들을 기준으로 키워드를 추출:
    - 공사종류, 공종, 사고객체, 작업프로세스, 장소
    """
    template = (
        "당신은 건설 안전 규정 및 사고 분석 전문가입니다. "
        "아래 문서는 건설 안전 지침서의 일부입니다. 문서 제목과 내용을 참고하여, "
        "사고 발생 시 중요한 다음 요소와 관련된 키워드를 추출해주세요. 공종은 공사종류입니다. 사고객체는 이 환경에서 사고가 났을 때의 물건입니다:\n"
        "- 공종\n"
        "- 사고객체\n"
        "- 작업프로세스\n"
        "- 장소\n\n"
        "각 카테고리별로 해당하는 키워드를 가장 중요한 순서대로 3개 제공해주세요. "
        "답변은 키워드로만 구성되어야 합니다.\n\n"
        "텍스트: {text}\n\n"
        "답변: "
    )
    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model="gemma3:27b")
    chain = prompt | model

    for idx, doc in enumerate(tqdm(documents, desc="Extracting LLM Keywords")):
        try:
            result = chain.invoke({"text": doc.page_content})
            doc.metadata["llm_keywords"] = result.strip()
            # 실시간으로 추출 결과 출력
            print(f"[Doc {idx+1}] Extracted Keywords: {doc.metadata['llm_keywords']}")
        except Exception as e:
            doc.metadata["llm_keywords"] = f"Error: {e}"
            print(f"[Doc {idx+1}] Error extracting keywords: {e}")
    return documents


# 4. FAISS 벡터 DB 구축 함수 (중간 결과 지속 출력)
def build_vectordb(data_dir: str):
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PDFPlumberLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from '{data_dir}'.")
    
    for idx, doc in enumerate(tqdm(documents, desc="Setting document titles")):
        if "source" in doc.metadata:
            doc.metadata["document_title"] = os.path.basename(doc.metadata["source"])
        else:
            doc.metadata["document_title"] = "Unknown Title"
        # 각 문서의 제목을 출력
        print(f"[Doc {idx+1}] Title set to: {doc.metadata['document_title']}")
    
    granite_embeddings = GraniteEmbeddings()
    
    # 문서를 의미 기반으로 청크 분할
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")
    
    # 첫 몇 개 청크의 미리보기 출력
    for i, doc in enumerate(docs[:5]):
        print(f"Chunk {i+1} preview: {doc.page_content[:200]}...")
    
    # LLM을 통한 키워드 추출 (실시간 출력 포함)
    docs = add_llm_keywords_to_documents(docs)
    
    vector_db = FAISS.from_documents(docs, granite_embeddings)
    print("Vector DB 구축 완료.")
    return vector_db


# 5. 벡터 DB 정보 출력 함수
def display_vectordb_info(vector_db, num_docs: int = 5):
    if hasattr(vector_db, "docstore") and hasattr(vector_db.docstore, "_dict"):
        docs = list(vector_db.docstore._dict.values())
    else:
        docs = None
    
    if docs is None:
        print("문서 정보를 찾을 수 없습니다.")
        return

    total_docs = len(docs)
    print(f"총 문서(청크) 수: {total_docs}")
    print("-" * 50)
    for i, doc in enumerate(docs[:num_docs]):
        title = doc.metadata.get("document_title", "No Title")
        keywords = doc.metadata.get("llm_keywords", "N/A")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"문서 {i+1}:")
        print("제목:", title)
        print("LLM 키워드 추출 결과:", keywords)
        print("내용 미리보기:", preview)
        print("-" * 50)


# 6. FAISS 벡터 DB 메타데이터를 CSV로 저장하는 함수
def save_vectordb_to_csv(vector_db, csv_path: str):
    if hasattr(vector_db, "docstore") and hasattr(vector_db.docstore, "_dict"):
        docs = list(vector_db.docstore._dict.values())
    else:
        print("Docstore에서 문서를 찾을 수 없습니다.")
        return

    records = []
    for doc in tqdm(docs, desc="Saving metadata to CSV"):
        record = {
            "document_title": doc.metadata.get("document_title", ""),
            "llm_keywords": doc.metadata.get("llm_keywords", ""),
            "page_content": doc.page_content
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"메타데이터가 CSV 파일로 저장되었습니다: {csv_path}")


if __name__ == "__main__":
    data_directory = "./"  # PDF 파일이 있는 폴더 경로
    vectordb = build_vectordb(data_directory)
    
    # 중간 결과를 출력하여 RAG DB에 어떤 데이터가 들어갔는지 확인
    display_vectordb_info(vectordb, num_docs=10)
    
    # FAISS 인덱스 저장
    faiss_save_dir = "faiss_index"
    vectordb.save_local(faiss_save_dir)
    print(f"FAISS 인덱스가 저장되었습니다: {faiss_save_dir}")
    
    # CSV 파일로 메타데이터 저장 (faiss_index 폴더 내에 metadata.csv 파일로 저장)
    csv_save_path = os.path.join(faiss_save_dir, "metadata.csv")
    save_vectordb_to_csv(vectordb, csv_save_path)
