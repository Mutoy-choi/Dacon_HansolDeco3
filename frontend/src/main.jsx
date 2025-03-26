import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";  // ✅ App.jsx를 불러옴
import "./index.css";  // ✅ 스타일 파일도 불러옴

// React를 <div id="root"></div> 안에 렌더링
ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
