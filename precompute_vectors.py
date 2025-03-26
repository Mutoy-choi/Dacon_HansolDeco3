import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# 1. --- CSV 데이터 로드 ---
train_data = pd.read_csv("./backend/database.csv")

# 2. --- NaN 값 제거 (결측값 있는 행 삭제) ---
factor_cols = ["사고원인", "공종", "작업프로세스", "인적사고", "물적사고", "장소", "사고객체", "부위"]
train_data = train_data.dropna(subset=factor_cols).reset_index(drop=True)

# 3. --- GPU 사용 여부 설정 ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# 4. --- SentenceTransformer 모델 로드 ---
model = SentenceTransformer("ibm-granite/granite-embedding-107m-multilingual", device=device)

# 5. --- 벡터화 수행 ---
vector_dict = {}  # 각 요소별 벡터 저장

for col in factor_cols:
    print(f"🔄 {col} 벡터화 진행 중...")
    vector_dict[col] = model.encode(train_data[col].tolist(), convert_to_numpy=True)

# 6. --- 벡터 데이터 저장 ---
np.savez("./backend/vectorized_data.npz", **vector_dict)

# 7. --- 정제된 데이터도 CSV로 저장 (필요할 경우) ---
train_data.to_csv("./backend/database.csv", index=False)

print(f"✅ 벡터 저장 완료: backend/vectorized_data.npz ({len(train_data)}개 데이터 사용)")
