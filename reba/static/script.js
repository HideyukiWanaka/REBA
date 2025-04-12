// static/script.js (最終版 - 全機能統合)

import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

// --- グローバル変数 ---
let poseLandmarker;
let runningMode = "VIDEO";
let webcamRunning = false;
let lastApiCallTime = 0;
const apiCallInterval = 500; // API呼び出し間隔 (ms)
let lastVideoTime = -1; // predictWebcam の重複実行防止用
let maxRebaScore = 0; // セッション中の最大REBAスコアを記録

// --- DOM要素 ---
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
// canvas要素が見つからない場合のエラーハンドリングを追加
if (!video || !canvasElement) {
    console.error("Fatal Error: Video or Canvas element not found in HTML!");
    alert("必要なHTML要素(video/canvas)が見つかりません。");
}
const canvasCtx = canvasElement ? canvasElement.getContext("2d") : null;
const drawingUtils = canvasCtx ? new DrawingUtils(canvasCtx) : null;
const scoreDisplay = document.getElementById("scoreDisplay");
const webcamButton = document.getElementById("webcamButton");

if (!scoreDisplay || !webcamButton) {
     console.error("Fatal Error: Score display or Webcam button element not found!");
     // 必要に応じてアラート表示など
}

// --- グラフ用変数 ---
let rebaChart = null; // Chart.js インスタンス (初期値 null)
const chartDataPoints = 60; // グラフに表示する最大データ点数
const chartData = {
  labels: [],
  datasets: [
    { label: 'REBA Total', data: [], borderColor: 'rgb(255, 99, 132)', backgroundColor: 'rgba(255, 99, 132, 0.1)', tension: 0.1, pointRadius: 0 },
    { label: 'Score A', data: [], borderColor: 'rgb(54, 162, 235)', backgroundColor: 'rgba(54, 162, 235, 0.1)', tension: 0.1, pointRadius: 0 },
    { label: 'Score B', data: [], borderColor: 'rgb(75, 192, 192)', backgroundColor: 'rgba(75, 192, 192, 0.1)', tension: 0.1, pointRadius: 0 }
  ]
};

/**
 * MediaPipe PoseLandmarkerを非同期で初期化 (Fullモデルを使用)
 */
async function initPoseLandmarker() {
  if (scoreDisplay) scoreDisplay.innerHTML = "<p>姿勢推定モデルの準備を開始...</p>";
  if (webcamButton) { webcamButton.disabled = true; webcamButton.innerText = "Loading..."; }
  console.log("Initializing PoseLandmarker...");
  try {
    if (scoreDisplay) scoreDisplay.innerHTML = "<p>実行ファイルをダウンロード中...</p>";
    console.log("Fetching vision tasks resolver...");
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    if (scoreDisplay) scoreDisplay.innerHTML = "<p>姿勢推定モデル(full)をダウンロード中...</p>";
    console.log("Resolver fetched. Creating PoseLandmarker (full)...");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO", numPoses: 1,
    });
    console.log("PoseLandmarker created successfully.");
    initChart(); // グラフ初期化呼び出し
    if (webcamButton) { webcamButton.disabled = false; webcamButton.innerText = "Recording Start"; }
    if (scoreDisplay) scoreDisplay.innerHTML = "モデル準備完了。ボタンを押して開始してください。";
  } catch (error) {
    console.error("Failed to initialize PoseLandmarker:", error);
    if (webcamButton) { webcamButton.disabled = true; webcamButton.innerText = "Load Failed"; }
    let errorMsg = `モデル初期化失敗: ${error.message}`;
    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError') || error instanceof TypeError) { errorMsg = "モデルのダウンロード失敗。ネットワーク確認要。"; }
    else if (error.message.includes('Wasm') || error.message.includes('WebGL')) { errorMsg = "ブラウザ/デバイスが未対応の可能性あり。"; }
    if (scoreDisplay) { scoreDisplay.innerHTML = `<p style="color: red;">エラー: ${errorMsg}</p>`; }
  }
}

/**
 * グラフを初期化
 */
function initChart() {
  if (!canvasCtx || !drawingUtils) { console.error("Cannot initialize chart: context not ready."); return; }
  if (rebaChart) { console.log("Chart already initialized."); return; }
  try {
    const chartCanvas = document.getElementById('rebaChart');
    if (!chartCanvas) { console.error("Chart canvas element 'rebaChart' not found."); return; }
    const ctx = chartCanvas.getContext('2d');
    if (!ctx) { console.error("Failed to get 2D context from chart canvas."); return; }

    rebaChart = new Chart(ctx, {
      type: 'line', data: chartData,
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, suggestedMax: 15, title: { display: true, text: 'Score' } },
          x: { ticks: { callback: function(v, i) { const N = Math.ceil(chartDataPoints / 10); return i % N === 0 ? this.getLabelForValue(v) : null; }, autoSkip: false, maxRotation: 0, minRotation: 0 }, title: { display: true, text: 'Time (Sequence)' } }
        },
        animation: { duration: 0 }, plugins: { legend: { position: 'bottom' }, title: { display: false } }
      }
    });
    console.log("Chart initialized successfully.");
  } catch(e) {
      console.error("Failed to initialize chart:", e);
      if(scoreDisplay) scoreDisplay.innerHTML += "<p style='color:red;'>グラフ初期化失敗</p>";
  }
}

/**
 * グラフを更新
 */
function updateChart(apiData) {
    if (!rebaChart) { console.warn("Chart not initialized, skipping update."); return; }
    if (!apiData) { console.warn("No apiData provided to updateChart, skipping."); return; }
    if (typeof apiData.intermediate_scores !== 'object' || apiData.intermediate_scores === null) {
        console.warn("apiData missing valid intermediate_scores, skipping score A/B update.", apiData);
    }

    try {
        const newLabel = chartData.labels.length > 0 ? Number(chartData.labels[chartData.labels.length - 1]) + 1 : 1;
        while (chartData.labels.length >= chartDataPoints) {
            chartData.labels.shift();
            chartData.datasets.forEach(dataset => { dataset.data.shift(); });
        }
        chartData.labels.push(newLabel);
        chartData.datasets[0].data.push(apiData.final_score ?? null);
        const scoreA = apiData.intermediate_scores?.scoreA ?? null;
        const scoreB = apiData.intermediate_scores?.scoreB ?? null;
        chartData.datasets[1].data.push(scoreA);
        chartData.datasets[2].data.push(scoreB);
        rebaChart.update();
    } catch(e) { console.error("Error during chart update:", e, apiData); }
}

/**
 * getUserMedia サポート確認
 */
function hasGetUserMedia() { return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia); }

/**
 * Webカメラを有効化 (環境カメラ要求)
 */
function enableCam() {
    if (!hasGetUserMedia()) { console.warn("getUserMedia not supported"); if(scoreDisplay) scoreDisplay.innerHTML = `<p style="color: red;">カメラ未対応</p>`; webcamRunning = false; if(webcamButton) webcamButton.innerText = "Recording Start"; return; }
    if (!poseLandmarker) { console.log("PoseLandmarker not loaded"); if(scoreDisplay) scoreDisplay.innerHTML = `<p>モデル未ロード</p>`; webcamRunning = false; if(webcamButton) webcamButton.innerText = "Recording Start"; return; }

    const constraints = { video: { facingMode: "environment" } }; // 環境カメラ要求
    console.log("Requesting camera with constraints:", constraints);
    navigator.mediaDevices.getUserMedia(constraints)
        .then((stream) => {
            if (!video) return;
            video.srcObject = stream;
            video.addEventListener("loadeddata", () => {
                if (!webcamRunning) { if (video.srcObject) { video.srcObject.getTracks().forEach(track => track.stop()); video.srcObject = null;} return; }
                if (!canvasElement) return;
                canvasElement.width = video.videoWidth;
                canvasElement.height = video.videoHeight;
                lastVideoTime = -1;
                requestAnimationFrame(predictWebcam); // ループ開始
            }, { once: true });
        })
        .catch((err) => {
            console.error("Error accessing webcam:", err);
            let userErrorMessage = `Webカメラエラー (${err.name})`;
            if (err.name === 'OverconstrainedError'){ userErrorMessage = `要求カメラ/設定未対応`; }
            else if (err.name === 'NotAllowedError') { userErrorMessage = `カメラ許可なし`; }
            else if (err.name === 'NotFoundError') { userErrorMessage = `カメラ未検出`; }
            if (scoreDisplay) scoreDisplay.innerHTML = `<p style="color: red;">${userErrorMessage}</p>`;
            webcamRunning = false; if(webcamButton) webcamButton.innerText = "Recording Start";
        });
 }

/**
 * REBAスコアからリスクレベル文字列を取得
 */
function getRiskLevelText(score) {
  if (score === null || score === undefined || score <= 0) return "N/A";
  if (score === 1) return "無視できる (Negligible)";
  if (score <= 3) return "低リスク (Low)";
  if (score <= 7) return "中リスク (Medium)";
  if (score <= 10) return "高リスク (High)";
  return "非常に高リスク (Very High)";
}

// --- ボタンのイベントリスナー (最大スコア関連あり) ---
if (webcamButton) {
    webcamButton.addEventListener("click", () => {
      if (!poseLandmarker && !webcamRunning) {
          console.log("Cannot start, model not loaded.");
          if(scoreDisplay) scoreDisplay.innerHTML = "<p style='color:red;'>モデル読込エラー</p>";
          return;
      }

      webcamRunning = !webcamRunning;
      webcamButton.innerText = webcamRunning ? "Stop Recording" : "Recording Start";

      if (webcamRunning) {
        // --- 開始時 ---
        maxRebaScore = 0; // 最大スコアをリセット
        if(scoreDisplay) scoreDisplay.innerHTML = "カメラを起動中...";
        // グラフデータをリセット (安全版)
        chartData.labels = [];
        chartData.datasets.forEach(dataset => { dataset.data = []; });
        if (rebaChart) { rebaChart.update(); }
        else { console.warn("Chart not initialized when trying to reset data on start."); }
        enableCam(); // カメラ起動
      } else {
        // --- 停止時 ---
        if (video && video.srcObject) { video.srcObject.getTracks().forEach(track => track.stop()); video.srcObject = null; console.log("Webcam stream stopped."); }
        if (canvasCtx) { canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height); }
        // 最大スコア表示
        const maxRiskLevel = getRiskLevelText(maxRebaScore);
        if (scoreDisplay) {
          if (maxRebaScore > 0) { scoreDisplay.innerHTML = `<h3>評価終了</h3><p>最大REBAスコア: <strong style="font-size: 1.2em;">${maxRebaScore}</strong></p><p>リスクレベル: <strong style="font-size: 1.1em;">${maxRiskLevel}</strong></p>`; }
          else { scoreDisplay.innerHTML = "評価停止中 (有効スコアなし)"; }
        }
        console.log(`Session stopped. Max REBA score was: ${maxRebaScore}`);
      }
    });
} else { console.error("Webcam button not found!"); }

// キャリブレーション入力取得 (堅牢版 + wristBaseScore マッピング修正版)
function getCalibrationInputs() {
    const names_in_html = [ "filmingSide", "neckRotation", "neckLateralBending", "trunkLateralFlexion", "loadForce", "shockForce", "postureCategory", "supportingLeg", "upperArmCorrection", "shoulderElevation", "gravityAssist", "wristCorrection", "wristAngleScore", "staticPosture", "repetitiveMovement", "unstableMovement", "coupling" ];
    const name_map = { "wristAngleScore": "wristBaseScore" }; // HTML name -> Python name
    const data = {};
    let errorOccurred = false;
    for (const html_name of names_in_html) {
        const python_name = name_map[html_name] || html_name;
        const element = document.querySelector(`input[name="${html_name}"]:checked`);
        if (element) {
            if (["filmingSide", "postureCategory", "supportingLeg"].includes(html_name)) { data[python_name] = element.value; }
            else { const value = Number(element.value); data[python_name] = isNaN(value) ? 0 : value; }
        } else {
            console.warn(`Could not find checked input for name="${html_name}". Assigning default.`); errorOccurred = true;
            if (["filmingSide", "postureCategory"].includes(python_name)) { data[python_name] = ""; }
            else if (python_name === "supportingLeg") { data[python_name] = null; }
            else if (python_name === "wristBaseScore") { data[python_name] = 1; }
            else { data[python_name] = 0; }
        }
    }
    if (errorOccurred) { console.warn("Errors occurred fetching some calibration inputs...", data); }
    else { /* console.log("Successfully obtained calibration inputs:", data); */ } // Reduce logs
    return data; // Return data even if incomplete
}

/**
 * メインループ (最大スコア更新処理あり)
 */
async function predictWebcam() {
  if (!webcamRunning || !poseLandmarker) { return; }
  if (!video || video.readyState < 2 || !canvasCtx || !drawingUtils) { if (webcamRunning) requestAnimationFrame(predictWebcam); return; }

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const startTimeMs = performance.now();

    poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
      if (!webcamRunning) return;

      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

      if (result.landmarks && result.landmarks.length > 0) {
        const landmarkSet = result.landmarks[0];
        try { // Draw landmarks safely
            drawingUtils.drawLandmarks(landmarkSet, { radius: (data) => DrawingUtils.lerp(data.from.z ?? 0, -0.15, 0.1, 5, 1) });
            drawingUtils.drawConnectors(landmarkSet, PoseLandmarker.POSE_CONNECTIONS);
        } catch(drawError) { console.error("Error drawing landmarks:", drawError); }

        // API スロットリング
        const now = performance.now();
        if (now - lastApiCallTime > apiCallInterval) {
          lastApiCallTime = now;
          const calibInputs = getCalibrationInputs();

          if (typeof calibInputs !== 'object' || calibInputs === null) {
              console.error("Skipping API call because calibInputs is invalid.", calibInputs);
              if(scoreDisplay && webcamRunning) { scoreDisplay.innerHTML = "<p style='color:red;'>エラー: 入力値取得失敗</p>"; }
              return; // Stop if inputs invalid
          }

          const payload = { landmarks: landmarkSet, calibInputs: calibInputs };

          let jsonPayload;
          try {
              jsonPayload = JSON.stringify(payload);
          } catch (stringifyError) {
              console.error("Error stringifying payload:", stringifyError, payload);
              if (scoreDisplay && webcamRunning) { scoreDisplay.innerHTML = `<p style="color: red;">エラー: 送信データ作成失敗</p>`; }
              return;
          }

          // API 呼び出しと Promise 処理 (修正版エラーハンドリング含む)
          const apiUrl = "https://reba-cgph.onrender.com/compute_reba"; // ★ 正しいURL ★
          // console.log("Calling API:", apiUrl); // Reduce logs

          fetch(apiUrl, { method: "POST", headers: { "Content-Type": "application/json" }, body: jsonPayload })
          .then(response => { // ① HTTP応答処理
              // console.log("[DEBUG] API response status:", response.status); // Reduce logs
              if (!response.ok) {
                  return response.text().then(text => {
                     console.error("[DEBUG] API error response body text:", text);
                     let errorMsg = `サーバーエラー Status: ${response.status}.`;
                     try {
                         const errData = JSON.parse(text);
                         if (response.status === 422 && errData.detail && Array.isArray(errData.detail)) { errorMsg = "データ検証エラー: " + errData.detail.map(e => `${e.loc?.join('.') || 'field'} - ${e.msg}`).join('; '); }
                         else if (errData.detail) { errorMsg = errData.detail; }
                         else { errorMsg += ` Response: ${text}`; }
                     } catch (e) { errorMsg += ` Response: ${text}`; }
                     throw new Error(errorMsg);
                  });
              }
              return response.json();
          })
          .then(data => { // ② 正常応答処理
             // console.log("[DEBUG] API success data object:", data); // Reduce logs
             if (!data) { throw new Error("API OK but data null/undefined."); }

             // 最大スコア更新
             if (typeof data.final_score === 'number' && data.final_score > maxRebaScore) { maxRebaScore = data.final_score; }
             // スコア表示更新
             if (scoreDisplay && webcamRunning) {
                 const score = (typeof data.final_score === 'number') ? data.final_score : 'N/A';
                 const risk = data?.risk_level ?? 'N/A';
                 scoreDisplay.innerHTML = `<p>最終REBAスコア: ${score}</p><p>リスクレベル: ${risk}</p>`;
             }
             // グラフ更新
             if (webcamRunning) { updateChart(data); }

             // ★★★ 計算された角度をコンソールに出力 ★★★
             if (data.computed_angles) {
                 console.log("Computed Angles:", data.computed_angles); // コンソールログは維持
             } else {
                 console.warn("Computed angles data missing in API response.");
             }
             // ★★★ ここまで角度出力 ★★★

          })
          .catch(err => { // ③ エラー処理
            console.error("[DEBUG] Error caught in fetch chain:", err);
            let displayMessage = err.message || "不明なAPIエラー";
            if (err.name === 'TypeError') { displayMessage = "API接続失敗"; }
            if (scoreDisplay && webcamRunning) { scoreDisplay.innerHTML = `<p style="color: red;">エラー: スコア取得失敗 (${displayMessage})</p>`; }
          });
        } // --- スロットリング終了 ---
      } else { /* console.log("No landmarks detected."); */ }
    }); // --- detectForVideo コールバック終了 ---
  } // --- video.readyState チェック終了 ---

  if (webcamRunning) { window.requestAnimationFrame(predictWebcam); }
} // --- predictWebcam 関数終了 ---

// アプリケーション初期化 (DOM読み込み後)
window.addEventListener('DOMContentLoaded', (event) => {
    console.log('DOM fully loaded and parsed');
    if(video && canvasElement && canvasCtx && drawingUtils && scoreDisplay && webcamButton){
        initPoseLandmarker(); // モデルとグラフの初期化を開始
    } else { console.error("Essential DOM elements missing!"); alert("ページ初期化失敗"); }
});