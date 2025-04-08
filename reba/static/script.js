// static/script.js (グラフ機能あり、最大スコア機能なしバージョン)

import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

// --- グローバル変数 ---
let poseLandmarker;
let runningMode = "VIDEO";
let webcamRunning = false;
let lastApiCallTime = 0;
const apiCallInterval = 500; // ms
let lastVideoTime = -1;

// --- DOM要素 ---
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);
const scoreDisplay = document.getElementById("scoreDisplay");
const webcamButton = document.getElementById("webcamButton");

// --- ★ グラフ用変数 ★ ---
let rebaChart = null;
const chartDataPoints = 60;
const chartData = {
  labels: [],
  datasets: [
    { label: 'REBA Total', data: [], borderColor: 'rgb(255, 99, 132)', backgroundColor: 'rgba(255, 99, 132, 0.1)', tension: 0.1, pointRadius: 0 },
    { label: 'Score A', data: [], borderColor: 'rgb(54, 162, 235)', backgroundColor: 'rgba(54, 162, 235, 0.1)', tension: 0.1, pointRadius: 0 },
    { label: 'Score B', data: [], borderColor: 'rgb(75, 192, 192)', backgroundColor: 'rgba(75, 192, 192, 0.1)', tension: 0.1, pointRadius: 0 }
  ]
};

/**
 * MediaPipe PoseLandmarkerを初期化 (Fullモデルを使用)
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
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO", numPoses: 1,
    });
    console.log("PoseLandmarker created successfully.");
    initChart(); // ★ グラフ初期化呼び出し ★
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
 * ★ グラフを初期化 ★
 */
function initChart() {
  if (rebaChart) { return; }
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
 * ★ グラフを更新 ★
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
    const constraints = { video: { facingMode: "environment" } }; // 環境カメラ
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
        .catch((err) => { /* ... (エラーハンドリング) ... */ });
 }

// --- ボタンのイベントリスナー (シンプルなバージョン) ---
webcamButton.addEventListener("click", () => {
  webcamRunning = !webcamRunning;
  webcamButton.innerText = webcamRunning ? "Stop Recording" : "Recording Start";
  if (webcamRunning) {
    if(scoreDisplay) scoreDisplay.innerHTML = "カメラを起動中...";
    enableCam();
  } else {
    if (video.srcObject) { video.srcObject.getTracks().forEach(track => track.stop()); video.srcObject = null; }
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (scoreDisplay) scoreDisplay.innerHTML = "評価停止中";
  }
});

// キャリブレーション入力取得
function getCalibrationInputs() { /* ... (内容は変更なし) ... */ }

/**
 * メインループ (グラフ更新呼び出しあり)
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
          const apiUrl = "https://reba-cgph.onrender.com/compute_reba"; // ★★★ 正しいバックエンドURL ★★★
          console.log("Calling API:", apiUrl);
          // predictWebcam 関数内の try ブロックの中
          try {
              const jsonPayload = JSON.stringify(payload);
              console.log("Stringified Payload for API:", jsonPayload);

              // ★★★ おそらくこの fetch と後続の .then/.catch の修正部分を指していました ★★★
              fetch(apiUrl, { method: "POST", headers: { "Content-Type": "application/json" }, body: jsonPayload })
              .then(response => {
                  console.log("API response status:", response.status);
                  if (!response.ok) { // 200番台以外のステータスコード(422など)の場合
                      // エラーレスポンスの本文を取得してエラーを投げる
                      return response.text().then(text => {
                         console.error("API error response body:", text);
                         let errorMsg = `サーバーエラー Status: ${response.status}. Response: ${text}`;
                         try {
                             // FastAPIの422エラー形式を解析試行
                             const errData = JSON.parse(text);
                             if (response.status === 422 && errData.detail && Array.isArray(errData.detail)) {
                                 errorMsg = "データ検証エラー: " + errData.detail.map(e => `${e.loc?.slice(-1)[0] || 'field'} - ${e.msg}`).join('; ');
                             } else if (errData.detail) {
                                 errorMsg = errData.detail; // 他のFastAPIエラー詳細
                             }
                         } catch (e) { /* JSONパース失敗時は元のテキストを使用 */ }
                         throw new Error(errorMsg); // ★ エラーをthrowしてcatchに渡す
                      });
                  }
                  return response.json(); // OKならJSONをパース
              })
              .then(data => { // ★ response.ok が true の場合のみここに来る ★
                 console.log("API success data:", data);
                 // スコア更新・グラフ更新など...
                 if (scoreDisplay && webcamRunning) { /* ... */ }
                 if (webcamRunning) { updateChart(data); }
              })
              .catch(err => { // ★ ネットワークエラーや上記でthrowされたエラーをここで捕捉 ★
                console.error("Full error object caught during API call:", err);
                let displayMessage = err.message || "不明なAPIエラー";
                // エラーメッセージ表示 (より具体的なエラーが出るはず)
                if (scoreDisplay && webcamRunning) {
                  scoreDisplay.innerHTML = `<p style="color: red;">エラー: REBAスコア取得失敗 (${displayMessage})</p>`;
                }
              });
              // ★★★ ここまでの一連の修正 ★★★
          } catch (stringifyError) { /* ... */ }
          } catch (stringifyError) { /* ... */ }
        } // --- スロットリング終了 ---
      } // --- ランドマーク処理終了 ---
    }); // --- detectForVideo コールバック終了 ---
  } // --- video.readyState チェック終了 ---
  if (webcamRunning) { window.requestAnimationFrame(predictWebcam); }
}

// アプリケーション初期化
initPoseLandmarker();
