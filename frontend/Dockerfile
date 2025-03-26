# Node.js 이미지 사용
FROM node:20

# 작업 디렉토리 설정
WORKDIR /app

# 패키지 설치
COPY package.json package-lock.json ./
RUN npm install

# 코드 복사 및 빌드
COPY . .
RUN npm run build

# Vite 개발 서버 실행
CMD ["npm", "run", "dev"]
