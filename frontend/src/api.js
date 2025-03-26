export async function generateResponses({ 사고객체, 사고원인, 공종, 작업프로세스, 장소, 부위, 인적사고, 물적사고 }) {
  const response = await fetch("http://localhost:8000/generate_responses", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      "사고객체": 사고객체,
      "사고원인": 사고원인,
      "공종": 공종,
      "작업프로세스": 작업프로세스,
      "장소": 장소,
      "부위": 부위,
      "인적사고": 인적사고,
      "물적사고": 물적사고,
    }),
  });

  if (!response.ok) {
    throw new Error(`API 오류 발생: ${response.statusText}`);
  }

  return response.json();
}

  
  export async function saveFeedback(query, winner, loser) {
    await fetch("http://localhost:8000/save_feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, winner, loser }),
    });
  }
  
  // 두 개의 응답을 병렬로 요청해서 하나의 결과로 반환
export async function generateResponsesTwice(inputData) {
  const [res1, res2] = await Promise.all([
    generateResponses(inputData),
    generateResponses(inputData),
  ]);

  return {
    query: res1.query, // 동일함
    top_cases: res1.top_cases, // 보통 첫 번째 기준 사용
    answer1: res1.answer,
    answer2: res2.answer,
  };
}