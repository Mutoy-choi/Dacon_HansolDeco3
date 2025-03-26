import React, { useState, useEffect } from "react";
import { generateResponses, saveFeedback } from "./api";
import "./index.css";
// generateResponsesTwice는 사용하지 않으므로 삭제하거나 추후에 필요시 사용할 수 있습니다.

export default function App() {
  // ✅ 상태 변수: 백엔드 QueryRequest에 맞춘 8개 요소
  const [accidentObject, setAccidentObject] = useState("");
  const [accidentCause, setAccidentCause] = useState("");
  const [gongjong, setGongjong] = useState("");
  const [jobProcess, setJobProcess] = useState("");
  const [location, setLocation] = useState("");
  const [part, setPart] = useState("");
  const [humanAccident, setHumanAccident] = useState("");
  const [materialAccident, setMaterialAccident] = useState("");

  const [answers, setAnswers] = useState(null);
  const [loading, setLoading] = useState(false);
  // ✅ 관련 문서 저장
  const [documents, setDocuments] = useState(null);
  // ✅  답변 편집 관련
  const [isEditing, setIsEditing] = useState(false);
  const [editedAnswer, setEditedAnswer] = useState("");
  const [selectedLoserAnswer, setSelectedLoserAnswer] = useState("");

  // ✅ 다음 테스트 케이스를 불러오는 함수
  const fetchNextTestCase = async () => {
    try {
      const response = await fetch("http://localhost:8000/next_test_case");
      const data = await response.json();
      if (data.message) {
        // 모든 테스트 케이스 소진 시
        alert(data.message);
        setAccidentObject("");
        setAccidentCause("");
        setGongjong("");
        setJobProcess("");
        setLocation("");
        setPart("");
        setHumanAccident("");
        setMaterialAccident("");
      } else {
        setAccidentObject(data.accident_object || "");
        setAccidentCause(data.accident_cause || "");
        setGongjong(data.gongjong || "");
        setJobProcess(data.jobProcess || "");
        setLocation(data.location || "");
        setPart(data.part || "");
        setHumanAccident(data.humanAccident || "");
        setMaterialAccident(data.materialAccident || "");
      }
      // 테스트 케이스가 바뀔 때마다 관련 문서 초기화
      setDocuments(null);
      setAnswers(null);
    } catch (error) {
      console.error("테스트 케이스 로드 오류:", error);
    }
  };

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const req = {
        사고객체: accidentObject,
        사고원인: accidentCause,
        공종: gongjong,
        작업프로세스: jobProcess,
        장소: location,
        부위: part,
        인적사고: humanAccident,
        물적사고: materialAccident,
      };

      // LLM 응답 2개를 병렬 호출
      const [answer1Data, answer2Data] = await Promise.all([
        generateResponses(req),
        generateResponses(req),
      ]);

      setAnswers({
        query: answer1Data.query, // 둘 다 동일
        top_cases: answer1Data.top_cases, // 유사 사례 (동일하거나 다를 수 있음)
        answer1: answer1Data.answer,
        answer2: answer2Data.answer,
      });
    } catch (error) {
      console.error("응답 생성 오류:", error);
    }
    setLoading(false);
  };

  // ✅ 피드백 저장 후 다음 테스트 케이스 로드
  const handleFeedback = async (winner, loser) => {
    try {
      await saveFeedback(answers.query, winner, loser);
      alert("피드백이 저장되었습니다!");
      setAnswers(null);
      fetchNextTestCase();
    } catch (error) {
      console.error("피드백 저장 오류:", error);
    }
  };

  // ✅ "/get_documents" API를 호출하여 관련 문서 가져오기
  const handleFetchDocuments = async () => {
    try {
      const response = await fetch(
        `http://localhost:8000/get_documents?accident_cause=${encodeURIComponent(accidentCause)}`
      );
      const data = await response.json();
      setDocuments(data);
    } catch (error) {
      console.error("문서 로드 오류:", error);
    }
  };

  // ✅ </think> 이후 줄 바꿈 처리 함수
  const formatResponse = (text) => {
    return text.replace(/<\/think>/g, "</think>\n\n");
  };
  
  // ✅ 답변 선택 시 편집 모드 전환 함수
  const handleSelectAnswer = (winner, loser) => {
    setEditedAnswer(winner);
    setSelectedLoserAnswer(loser);
    setIsEditing(true);
  };

  // ✅ 수정 완료 후 피드백 제출 함수
  const handleSaveEditedAnswer = () => {
    // 수정된 답변을 winner로, 나머지 답변을 loser로 전달
    handleFeedback(editedAnswer, selectedLoserAnswer);
  };

  // 컴포넌트 마운트 시 첫 테스트 케이스 로드
  useEffect(() => {
    fetchNextTestCase();
  }, []);

  return (
    <div className="min-h-screen flex flex-col items-center bg-gray-100 p-4 sm:p-6">
      <div className="w-full max-w-3xl bg-white shadow-md p-4 sm:p-6 rounded-lg">
        <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-center text-blue-700 mb-4 md:mb-6">
          건설 사고 대응 시스템
        </h1>

        {/* 입력 필드 */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            🛠 사고 객체
          </label>
          <input
            type="text"
            className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-400"
            value={accidentObject}
            onChange={(e) => setAccidentObject(e.target.value)}
          />
        </div>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            ⚠️ 사고 원인
          </label>
          <input
            type="text"
            className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-400"
            value={accidentCause}
            onChange={(e) => setAccidentCause(e.target.value)}
          />
        </div>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            🏗 공종
          </label>
          <input
            type="text"
            className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-400"
            value={gongjong}
            onChange={(e) => setGongjong(e.target.value)}
          />
        </div>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            🔄 작업프로세스
          </label>
          <input
            type="text"
            className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-400"
            value={jobProcess}
            onChange={(e) => setJobProcess(e.target.value)}
          />
        </div>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            📍 장소
          </label>
          <input
            type="text"
            className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-400"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
          />
        </div>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            🔖 부위
          </label>
          <input
            type="text"
            className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-400"
            value={part}
            onChange={(e) => setPart(e.target.value)}
          />
        </div>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            👥 인적 사고
          </label>
          <input
            type="text"
            className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-400"
            value={humanAccident}
            onChange={(e) => setHumanAccident(e.target.value)}
          />
        </div>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            💥 물적 사고
          </label>
          <input
            type="text"
            className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-400"
            value={materialAccident}
            onChange={(e) => setMaterialAccident(e.target.value)}
          />
        </div>

        {/* 응답 생성 및 관련 문서 버튼 */}
        <button
          className={`w-full p-3 rounded-lg text-white font-semibold ${
            loading || !accidentObject || !accidentCause
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700"
          }`}
          onClick={handleGenerate}
          disabled={loading || !accidentObject || !accidentCause}
        >
          {loading ? "생성 중..." : "🚀 답변 생성"}
        </button>
        <button
          className="mt-4 w-full p-3 rounded-lg text-white font-semibold bg-purple-600 hover:bg-purple-700"
          onClick={handleFetchDocuments}
        >
          📄 관련 문서 가져오기
        </button>

        {/* LLM 답변 영역 */}
        {answers && (
          <div className="mt-6">
            <h2 className="text-lg font-semibold text-gray-700 mb-2">
              📌 입력 프롬프트
            </h2>
            <div className="p-3 bg-gray-100 rounded-md text-gray-800 whitespace-pre-wrap">
              {JSON.stringify(answers.query, null, 2)}
            </div>

            <h2 className="text-lg font-semibold mt-4 text-gray-700 mb-2">
              🔍 유사 사례
            </h2>
            <div className="p-3 bg-gray-100 rounded-md text-gray-800">
              {answers.top_cases.map((caseData, index) => (
                <div key={index} className="mb-3 border-b pb-2">
                  <p className="font-semibold">
                    {index + 1}. (유사도:{" "}
                    {Math.round(caseData.similarity * 100)}%)
                  </p>
                  <p className="text-sm">
                    🛠 사고객체: {caseData["사고객체"]}
                  </p>
                  <p className="text-sm">
                    ⚠️ 사고원인: {caseData["사고원인"]}
                  </p>
                  <p className="text-sm">
                    ✅ 대응 대책: {caseData["재발방지대책 및 향후조치계획"]}
                  </p>
                </div>
              ))}
            </div>

            <h2 className="text-lg font-semibold mt-6 text-gray-700 mb-2">
              💬 LLM 답변 (더 나은 답변을 선택하세요)
            </h2>
            {isEditing ? (
              <div className="mt-4">
                <h3 className="text-lg font-semibold text-gray-700 mb-2">
                  선택한 답변 수정하기
                </h3>
                <textarea
                  className="w-full p-3 border rounded-md"
                  rows={6}
                  value={editedAnswer}
                  onChange={(e) => setEditedAnswer(e.target.value)}
                />
                <div className="mt-2 flex gap-2">
                  <button
                    className="px-4 py-2 bg-blue-500 text-white rounded-md"
                    onClick={handleSaveEditedAnswer}
                  >
                    수정 완료 및 제출
                  </button>
                  <button
                    className="px-4 py-2 bg-gray-300 text-gray-800 rounded-md"
                    onClick={() => setIsEditing(false)}
                  >
                    취소
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex flex-row gap-4">
                <button
                  className="flex-1 p-4 border rounded-lg bg-white shadow-md hover:shadow-lg transition"
                  onClick={() =>
                    handleSelectAnswer(answers.answer1, answers.answer2)
                  }
                >
                  <div className="whitespace-pre-wrap break-words">
                    {formatResponse(answers.answer1)}
                  </div>
                </button>
                <button
                  className="flex-1 p-4 border rounded-lg bg-white shadow-md hover:shadow-lg transition"
                  onClick={() =>
                    handleSelectAnswer(answers.answer2, answers.answer1)
                  }
                >
                  <div className="whitespace-pre-wrap break-words">
                    {formatResponse(answers.answer2)}
                  </div>
                </button>
              </div>
            )}
          </div>
        )}

        {/* 구분선: LLM 답변 영역과 관련 문서 영역을 명확히 분리 */}
        {documents && <hr className="my-6 border-gray-300" />}

        {/* 관련 문서 영역 */}
        {documents && (
          <div className="mt-6">
            <h2 className="text-lg font-semibold text-gray-700 mb-2">
              📄 관련 문서
            </h2>
            <div className="p-3 bg-gray-100 rounded-md text-gray-800">
              {documents.documents && documents.documents.length > 0 ? (
                documents.documents.map((doc, index) => (
                  <div key={index} className="mb-3 border-b pb-2">
                    <p className="font-semibold">제목: {doc.title}</p>
                    <p className="text-sm">
                      LLM 키워드: {doc.llm_keywords}
                    </p>
                    <p className="text-sm">
                      유사도: {(doc.llm_keywords_similarity * 100).toFixed(1)}%
                    </p>
                    <p className="text-sm">문서 내용: {doc.chunk_content}</p>
                  </div>
                ))
              ) : (
                <p>유사도 임계치를 만족하는 문서를 찾지 못했습니다.</p>
              )}
            </div>
          </div>
        )}

        {/* 수동으로 다음 테스트 케이스 불러오기 버튼 */}
        <button
          className="mt-4 w-full p-3 rounded-lg text-white font-semibold bg-green-600 hover:bg-green-700"
          onClick={fetchNextTestCase}
        >
          다음 테스트 케이스 불러오기
        </button>
      </div>
    </div>
  );
}
