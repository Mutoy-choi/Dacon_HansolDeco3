# 🚀 LLM Feedback System (RAG 기반 사고 대응 시스템)

**목차**  
1. [프로젝트 개요](#프로젝트-개요)  
2. [구성 요소](#구성-요소)  
3. [설치 및 실행 방법](#설치-및-실행-방법)  
   - [Docker Compose로 실행하기](#docker-compose로-실행하기)  
   - [로컬 환경에서 직접 실행하기](#로컬-환경에서-직접-실행하기)  
4. [주요 기능](#주요-기능)  
5. [파일 구조](#파일-구조)  
6. [라이선스](#라이선스)

---

## 프로젝트 개요
이 프로젝트는 RAG(검색 결합 생성, Retrieval-Augmented Generation) 방식을 사용하여 **사고 객체와 사고 원인을 입력**하면, **과거 유사한 사고 사례**를 검색한 뒤 **LLM이 두 가지 대응 방안을 생성**합니다. 이후 사용자가 **더 나은 답변을 선택하거나**, 혹은 직접 답안을 수정해 저장하면 이를 **데이터로 축적**하여 향후 모델을 개선(RLHF와 유사한 학습)할 수 있습니다. 또한, 건설안전지침 등 **관련 문서를 조회**할 수 있어, 답안을 보완하는 데 활용할 수 있습니다.

---

## 데모 시연

###  LLM 기반 대응 방안 생성
- Ollama(로컬 LLM)에 질의하여 **2가지 대응 방안** 생성
- 생성된 결과를 프론트엔드에 전달
![답변 생성 데모](./demo_gif/답변생성.gif)

---

### 사용자 선택 및 수정
- 프론트엔드에서 사용자에게 두 가지 대응 방안을 모두 보여줌  
- 사용자는 **더 나은 답변**을 선택하거나, 직접 **답변을 편집** 가능
![답변 수정 데모](./demo_gif/답변수정.gif)

---

### 건설안전지침 등 문서 조회
- 사용자가 궁금한 사항을 확인할 수 있도록 관련 **건설안전지침** 문서를 연동
![문서 검색 데모](./demo_gif/관련문서가져오기.gif)

---

## 구성 요소
1. **백엔드(backend)**
   - **프레임워크**: [FastAPI](https://fastapi.tiangolo.com/)
   - **포트**: 8000
   - **역할**: 
     - 문서 벡터화(사전에 `ibm-granite/granite-embedding-107m-multilingual`로 임베딩 후 DB 또는 파일로 관리)
     - 사용자가 입력한 사고 정보와 유사도 검색
     - LLM을 통해 두 가지 대응 방안 생성
     - 사용자가 선택·수정한 결과를 저장
   - **사전 처리 스크립트**: `precompute_vectors.py` (유사도 검색용 벡터 생성)

2. **프론트엔드(frontend)**
   - **프레임워크**: [React](https://reactjs.org/)
   - **포트**: 3000
   - **역할**: 
     - 사용자 입력(사고 객체, 원인 등) 인터페이스
     - 생성된 두 가지 대응 방안 중 선택 및 편집
     - 건설안전지침(참고 문서) 확인 기능

3. **LLM(Ollama)**
   - **이미지**: [`ollama/ollama`](https://ollama.com)
   - **역할**:  
     - 로컬 환경에서 LLM 사용
     - 백엔드가 LLM 요청 시 실행
   - **모델 다운로드 및 캐시 경로**: `/root/.ollama` (Docker 볼륨을 통해 영구 보관 가능)

---

## 설치 및 실행 방법

## 데이터 준비

LLM의 유사도 검색 및 대응 방안 생성을 위한 벡터 데이터가 필요합니다. 아래 두 가지 방법 중 하나를 선택하세요.

### 1. Google Drive 파일 다운로드

아래 링크에서 필요한 파일을 다운로드한 후, **backend** 폴더에 넣어주세요.

[다운로드 링크](https://drive.google.com/file/d/1TgNFsKpii-SGkkxd6Rd7MLXWJuJ2NZQn/view?usp=sharing)

커맨드 라인에서 직접 다운로드하고 싶다면, 아래의 `curl` 명령어를 사용해보세요 (파일 이름은 원하는 이름으로 수정):

```bash
curl -L "https://drive.google.com/uc?export=download&id=1TgNFsKpii-SGkkxd6Rd7MLXWJuJ2NZQn" -o backend/vectorized_data.npz
```

### 2. precompute_vectors.py 실행

다운로드한 데이터나 기존 데이터로부터 벡터를 생성하고 싶다면, **backend** 폴더에서 아래 명령어를 실행하세요:

```bash
cd backend
python precompute_vectors.py
```

이 스크립트는 유사도 검색에 필요한 벡터 데이터를 생성합니다.

### Docker Compose로 실행하기

#### 1. Docker & Docker Compose 설치
- [Docker](https://docs.docker.com/engine/install/) 및 [Docker Compose](https://docs.docker.com/compose/install/)가 설치되어 있어야 합니다.

#### 2. 프로젝트 빌드 및 실행
```bash
# 프로젝트 루트 디렉터리에서 실행
docker-compose up --build
```
- `--build` 옵션은 컨테이너 빌드 후 실행합니다.
- 정상적으로 실행되면 프론트엔드는 `http://localhost:3000`, 백엔드는 `http://localhost:8000`에서 접근 가능합니다.

#### 3. 종료
```bash
docker-compose down
```
- 실행 중인 모든 컨테이너가 종료됩니다.

---

### 로컬 환경에서 직접 실행하기

Docker를 사용하지 않고, 로컬 환경에서 직접 FastAPI와 React를 실행할 수도 있습니다.

#### 1. 백엔드 실행
```bash
# 1) 백엔드 디렉터리로 이동
cd backend

# 2) 필요 패키지 설치
pip install -r requirements.txt

# 3) 벡터 사전 생성(사고 사례 등 문서 벡터화)
python precompute_vectors.py

# 4) FastAPI 실행
uvicorn main:app --reload
# => http://localhost:8000 에서 API 사용 가능
```

#### 2. 프론트엔드 실행
```bash
# 1) 프론트엔드 디렉터리로 이동
cd frontend

# 2) 패키지 설치
npm install

# 3) 개발 서버 실행
npm run start
# => http://localhost:3000 에서 프론트엔드 확인 가능
```

#### 3. Ollama(LLM) 실행
- Ollama 모델을 로컬에서 사용하기 위해서는 [공식 문서](https://ollama.com) 참조 후 설치가 필요합니다.
- 설치 후, 특정 모델(qwq)을 다운로드하세요.:
  ```bash
  ollama pull qwq
  ```
- 로컬 환경에서 모델이 구동 중이라면, FastAPI에서 Ollama API를 호출하여 답변을 생성할 수 있습니다.

---

## 주요 기능

1. **사고 정보 입력**  
   - 사고 객체(사람, 장비, 환경 등)와 사고 원인 등을 입력받습니다.

2. **유사 사고 사례 검색**  
   - `Sentence-BERT` 등의 임베딩을 사전에 계산(`precompute_vectors.py`)  
   - 사용자가 입력한 정보와 유사도가 높은 사고 사례(문서) 추출  

3. **LLM 기반 대응 방안 생성**  
   - Ollama(로컬 LLM)에 질의하여 **2가지 대응 방안** 생성  
   - 생성된 결과를 프론트엔드에 전달  

4. **사용자 선택 및 수정**  
   - 프론트엔드에서 사용자에게 두 가지 대응 방안을 모두 보여줌  
   - 사용자는 **더 나은 답변**을 선택하거나, 직접 **답변을 편집** 가능  

5. **데이터 저장 (RLHF-like 강화학습용)**  
   - 사용자 피드백(선택 및 수정한 텍스트)은 별도의 DB 혹은 파일로 저장  
   - 이후 학습에 반영 가능  

6. **건설안전지침 등 문서 조회**  
   - 사용자가 궁금한 사항을 확인할 수 있도록 관련 **건설안전지침** 문서를 연동  
   - 프론트엔드에서 버튼 클릭 시 열람 가능  

---

