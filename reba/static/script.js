// static/script.js (最終版)

import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

// --- グローバル変数 ---
let poseLandmarker;
let runningMode = "VIDEO";
let webcamRunning = false;
let lastApiCallTime = 0;
const apiCallInterval = 500; // API呼び出し間隔 (ms)
let lastVideoTime = -1; // predictWebcam の重複実行防止用
let maxRebaScore = 0; // ★ セッション中の最大REBAスコアを記録 ★

// --- DOM要素 ---
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);
const scoreDisplay = document.getElementById("scoreDisplay");
const webcamButton = document.getElementById("webcamButton");

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
  webcamButton.disabled = true; webcamButton.innerText = "Loading...";
  console.log("Initializing PoseLandmarker...");
  try {
    if (scoreDisplay) scoreDisplay.innerHTML = "<p>実行ファイルをダウンロード中...</p>";
    console.log("Fetching vision tasks resolver...");
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    if (scoreDisplay) scoreDisplay.innerHTML = "<p>姿勢推定モデル(full)をダウンロード中...</p>";
    console.log("Resolver fetched. Creating PoseLandmarker (full)...");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        // Fullモデルのパスを使用
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        delegate: "GPU" // または "CPU"
      },
      runningMode: "VIDEO", numPoses: 1,
    });
    console.log("PoseLandmarker created successfully.");
    initChart(); // グラフ初期化呼び出し
    webcamButton.disabled = false; webcamButton.innerText = "Recording Start";
    if (scoreDisplay) scoreDisplay.innerHTML = "モデル準備完了。ボタンを押して開始してください。";
  } catch (error) {
    console.error("Failed to initialize PoseLandmarker:", error);
    webcamButton.disabled = true; webcamButton.innerText = "Load Failed";
    let errorMsg = `モデル初期化失敗: ${error.message}`;
    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError') || error instanceof TypeError) { errorMsg = "モデルのダウンロードに失敗しました..."; }
    else if (error.message.includes('Wasm') || error.message.includes('WebGL')) { errorMsg = "ブラウザ/デバイスがモデル実行機能をサポートしていません..."; }
    if (scoreDisplay) { scoreDisplay.innerHTML = `<p style="color: red;">エラー: ${errorMsg}</p>`; }
  }
}

/**
 * グラフを初期化
 */
function initChart() {
  if (rebaChart) { return; } // 二重初期化防止
  try {
    const ctx = document.getElementById('rebaChart').getContext('2d');
    if (!ctx) { console.error("Chart canvas element 'rebaChart' not found."); return; }
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
      if(scoreDisplay) scoreDisplay.innerHTML += "<p style='color:red;'>グラフの初期化に失敗しました。</p>";
  }
}

/**
 * グラフを更新
 * @param {object} apiData APIからのレスポンスデータ
 */
function updateChart(apiData) {
    if (!rebaChart || !apiData || !apiData.intermediate_scores) { console.warn("Chart update skipped.", {chart: !!rebaChart, data: apiData}); return; }
    try {
        const newLabel = chartData.labels.length > 0 ? Number(chartData.labels[chartData.labels.length - 1]) + 1 : 1;
        while (chartData.labels.length >= chartDataPoints) {
            chartData.labels.shift();
            chartData.datasets.forEach(dataset => { dataset.data.shift(); });
        }
        chartData.labels.push(newLabel);
        chartData.datasets[0].data.push(apiData.final_score ?? null);
        chartData.datasets[1].data.push(apiData.intermediate_scores.scoreA ?? null);
        chartData.datasets[2].data.push(apiData.intermediate_scores.scoreB ?? null);
        rebaChart.update();
    } catch(e) { console.error("Failed to update chart data:", e, apiData); }
}

/**
 * getUserMedia サポート確認
 */
function hasGetUserMedia() { return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia); }

/**
 * Webカメラを有効化 (環境カメラ要求)
 */
function enableCam() {
    if (!hasGetUserMedia()) { /*...*/ return; }
    if (!poseLandmarker) { /*...*/ return; }
    // ★ 環境カメラを要求 ★
    const constraints = { video: { facingMode: "environment" } };
    console.log("Requesting camera with constraints:", constraints);
    navigator.mediaDevices.getUserMedia(constraints)
        .then((stream) => {
            video.srcObject = stream;
            video.addEventListener("loadeddata", () => {
                canvasElement.width = video.videoWidth;
                canvasElement.height = video.videoHeight;
                lastVideoTime = -1;
                if (webcamRunning) { requestAnimationFrame(predictWebcam); }
            }, { once: true });
        })
        .catch((err) => { /* ... (エラーハンドリング) ... */
            console.error("Error accessing webcam:", err);
            let userErrorMessage = `Webカメラアクセスエラー (${err.name}: ${err.message})`;
            if (err.name === 'OverconstrainedError') { userErrorMessage = `要求されたカメラ設定(環境カメラ等)がサポートされていません...`; }
            else if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') { userErrorMessage = `カメラへのアクセスが許可されませんでした...`; }
            else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') { userErrorMessage = `利用可能なカメラが見つかりませんでした。`; }
            if (scoreDisplay) scoreDisplay.innerHTML = `<p style="color: red;">${userErrorMessage}</p>`;
            webcamRunning = false; webcamButton.innerText = "Recording Start";
        });
 }

/**
 * ★ REBAスコアからリスクレベル文字列を取得 ★
 */
function getRiskLevelText(score) {
  if (score === null || score === undefined || score <= 0) return "N/A";
  if (score === 1) return "無視できる (Negligible)";
  if (score <= 3) return "低リスク (Low)";
  if (score <= 7) return "中リスク (Medium)";
  if (score <= 10) return "高リスク (High)";
  return "非常に高リスク (Very High)";
}

// --- ★ ボタンのイベントリスナー (最大スコア関連あり) ★ ---
webcamButton.addEventListener("click", () => {
  webcamRunning = !webcamRunning;
  webcamButton.innerText = webcamRunning ? "Stop Recording" : "Recording Start";

  if (webcamRunning) {
    // --- 開始時 ---
    maxRebaScore = 0; // ★ 最大スコアをリセット ★
    if(scoreDisplay) scoreDisplay.innerHTML = "カメラを起動中...";
    // ★ グラフデータのリセット (安全版) ★
    chartData.labels = [];
    chartData.datasets.forEach(dataset => { dataset.data = []; });
    if (rebaChart) { rebaChart.update(); } // グラフオブジェクトがあれば更新

    enableCam();
  } else {
    // --- 停止時 ---
    if (video.srcObject) { video.srcObject.getTracks().forEach(track => track.stop()); video.srcObject = null; console.log("Webcam stream stopped."); }
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // ★ 最大スコアとリスクレベルを表示 ★
    const maxRiskLevel = getRiskLevelText(maxRebaScore);
    if (scoreDisplay) {
      if (maxRebaScore > 0) {
        scoreDisplay.innerHTML =
          `<h3>評価終了</h3>
           <p>今回の最大REBAスコア: <strong style="font-size: 1.2em;">${maxRebaScore}</strong></p>
           <p>対応リスクレベル: <strong style="font-size: 1.1em;">${maxRiskLevel}</strong></p>`;
      } else {
        scoreDisplay.innerHTML = "評価停止中 (有効なスコアなし)";
      }
    }
    console.log(`Session stopped. Max REBA score was: ${maxRebaScore}`);
  }
});

// キャリブレーション入力取得 (堅牢版)
function getCalibrationInputs() {
    const names = [ "filmingSide", "neckRotation", "neckLateralBending", "trunkLateralFlexion", "loadForce", "postureCategory", "upperArmCorrection", "shoulderElevation", "gravityAssist", "wristCorrection", "staticPosture", "repetitiveMovement", "unstableMovement", "coupling" ];
    const data = {};
    let errorOccurred = false;
    for (const name of names) {
        const element = document.querySelector(`input[name="${name}"]:checked`);
        if (element) {
            if (name === "filmingSide" || name === "postureCategory") { data[name] = element.value; }
            else { const value = Number(element.value); data[name] = isNaN(value) ? 0 : value; }
        } else {
            console.warn(`Could not find checked input for name="${name}". Assigning default/null.`);
            errorOccurred = true;
            data[name] = (name === "filmingSide" || name === "postureCategory") ? "" : 0;
        }
    }
    if (errorOccurred) { console.warn("Errors occurred fetching some calibration inputs. Data might be incomplete:", data); }
    else { console.log("Successfully obtained calibration inputs:", data); }
    return data;
}


/**
 * メインループ (最大スコア更新処理あり)
 */
async function predictWebcam() {
  if (!webcamRunning) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const startTimeMs = performance.now();
    poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
      if (!webcamRunning) return;
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      if (result.landmarks && result.landmarks.length > 0) {
        const landmarkSet = result.landmarks[0];
        drawingUtils.drawLandmarks(landmarkSet, { /* ... */ });
        drawingUtils.drawConnectors(landmarkSet, PoseLandmarker.POSE_CONNECTIONS);
        // API スロットリング
        const now = performance.now();
        if (now - lastApiCallTime > apiCallInterval) {
          lastApiCallTime = now;
          const calibInputs = getCalibrationInputs();
          const payload = { landmarks: landmarkSet, calibInputs: calibInputs };
          // ★★★ 正しいバックエンドURL ★★★
          const apiUrl = "https://reba-cgph.onrender.com/compute_reba";
          console.log("Calling API:", apiUrl);

          if (typeof calibInputs === 'undefined' || calibInputs === null) { // calibInputsチェック強化
              console.error("Skipping API call because calibInputs is invalid.", calibInputs);
              if(scoreDisplay && webcamRunning) { scoreDisplay.innerHTML = "<p style='color:red;'>エラー: 入力値取得失敗</p>"; }
              return; // ストップ
          }

          try {
              const jsonPayload = JSON.stringify(payload);
              fetch(apiUrl, { method: "POST", headers: { "Content-Type": "application/json" }, body: jsonPayload })
              .then(response => { // レスポンス処理 (変更なし)
                    console.log("API response status:", response.status);
                    if (!response.ok) { return response.text().then(text => { /* ... エラーthrow ... */ }); }
                    return response.json();
               })
              .then(data => { // 正常系データ処理
                 console.log("API success data:", data);
                 // ★ 最大スコア更新 ★
                 if (data && data.final_score !== null && data.final_score !== undefined && data.final_score > maxRebaScore) {
                   maxRebaScore = data.final_score;
                   console.log(`New max REBA score recorded: ${maxRebaScore}`);
                 }
                 // 現在スコア表示更新
                 if (scoreDisplay && webcamRunning) {
                   const score = data?.final_score ?? 'N/A';
                   const risk = data?.risk_level ?? 'N/A';
                   scoreDisplay.innerHTML = `<p>最終REBAスコア: ${score}</p><p>リスクレベル: ${risk}</p>`;
                 }
                 // グラフ更新
                 if (webcamRunning) { updateChart(data); }
              })
              .catch(err => { // fetch Promise チェーンのエラー処理
                 console.error("Full error object caught during API call:", err);
                 let displayMessage = err.message || "不明なAPIエラー";
                 if (err.name === 'TypeError') { displayMessage = "API接続失敗。URL/ネットワーク確認"; }
                 if (scoreDisplay && webcamRunning) { scoreDisplay.innerHTML = `<p style="color: red;">エラー: REBAスコア取得失敗 (${displayMessage})</p>`; }
              });
          } catch (stringifyError) { // 同期エラー処理
              console.error("Error stringifying payload:", stringifyError, payload);
              if (scoreDisplay && webcamRunning) { scoreDisplay.innerHTML = `<p style="color: red;">エラー: 送信データ作成失敗</p>`; }
          }
        } // --- スロットリング終了 ---
      } // --- ランドマーク処理終了 ---
    }); // --- detectForVideo コールバック終了 ---
  } // --- video.readyState チェック終了 ---
  if (webcamRunning) { window.requestAnimationFrame(predictWebcam); }
}

// アプリケーション初期化
initPoseLandmarker();
