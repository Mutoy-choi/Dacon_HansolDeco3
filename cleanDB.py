import pandas as pd
import re

# CSV 파일 불러오기
df = pd.read_csv("faiss_index/metadata.csv")

# llm_keywords 정제 함수
def clean_llm_keywords(text):
    return re.sub(r"\*\*", "", text)

# page_content 정제 함수
def clean_page_content(text):
    text = re.sub(r"\.pdf", "", text)
    text = re.sub(r"- \d+ -", "", text).strip()
    return text

# 각 컬럼에 함수 적용
df["llm_keywords"] = df["llm_keywords"].apply(clean_llm_keywords)
df["page_content"] = df["page_content"].apply(clean_page_content)

# 정제된 데이터 CSV로 저장
df.to_csv("faiss_index/metadata.csv", index=False)

print("✅ CSV 파일의 데이터가 정제되어 저장되었습니다!")
