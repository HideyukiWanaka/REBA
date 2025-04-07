// static/script.js (修正版)

import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

// --- グローバル変数 (変更なし) ---
let poseLandmarker;
let runningMode = "VIDEO";
let webcamRunning = false;
let lastApiCallTime = 0;
const apiCallInterval = 500;
let lastVideoTime = -1;

// --- DOM要素 (変更なし) ---
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);
const scoreDisplay = document.getElementById("scoreDisplay");
const webcamButton = document.getElementById("webcamButton");

// --- グラフ用変数 (変更なし) ---
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
  webcamButton.disabled = true;
  webcamButton.innerText = "Loading...";

  console.log("Initializing PoseLandmarker...");
  try {
    if (scoreDisplay) scoreDisplay.innerHTML = "<p>実行ファイルをダウンロード中...</p>";
    console.log("Fetching vision tasks resolver...");
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");

    if (scoreDisplay) scoreDisplay.innerHTML = "<p>姿勢推定モデル(full)をダウンロード中...</p>"; // Fullモデル使用を明記
    console.log("Resolver fetched. Creating PoseLandmarker (full)...");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        // ★ Fullモデルのパスを使用 ★
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
    console.log("PoseLandmarker created successfully.");
    initChart(); // グラフ初期化
    webcamButton.disabled = false;
    webcamButton.innerText = "Recording Start";
    if (scoreDisplay) scoreDisplay.innerHTML = "モデル準備完了。ボタンを押して開始してください。";
  } catch (error) {
    console.error("Failed to initialize PoseLandmarker:", error);
    webcamButton.disabled = true;
    webcamButton.innerText = "Load Failed";
    let errorMsg = `モデル初期化失敗: ${error.message}`;
    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError') || error instanceof TypeError) {
        errorMsg = "モデルのダウンロードに失敗しました。ネットワーク接続を確認してください。";
    } else if (error.message.includes('Wasm') || error.message.includes('WebGL')) {
        errorMsg = "ブラウザ/デバイスがモデル実行機能をサポートしていない可能性があります。";
    }
    if (scoreDisplay) {
        scoreDisplay.innerHTML = `<p style="color: red;">エラー: ${errorMsg}</p>`;
    }
  }
}

/**
 * グラフを初期化
 */
function initChart() {
    // ... (グラフ初期化コードは変更なし) ...
    try {
        const ctx = document.getElementById('rebaChart').getContext('2d');
        if (!ctx) { console.error("Chart canvas element 'rebaChart' not found."); return; }
        rebaChart = new Chart(ctx, {
        type: 'line', data: chartData,
        options: { responsive: true, maintainAspectRatio: false,
            scales: {
            y: { beginAtZero: true, suggestedMax: 15, title: { display: true, text: 'Score' } },
            x: { ticks: { callback: function(value, index, values) { const N = Math.ceil(chartDataPoints / 10); return index % N === 0 ? this.getLabelForValue(value) : null; }, autoSkip: false, maxRotation: 0, minRotation: 0 }, title: { display: true, text: 'Time (Sequence)' } }
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
    // ... (グラフ更新コードは変更なし) ...
    if (!rebaChart || !apiData || !apiData.intermediate_scores) { return; }
    try {
        const newLabel = chartData.labels.length > 0 ? Number(chartData.labels[chartData.labels.length - 1]) + 1 : 1;
        while (chartData.labels.length >= chartDataPoints) {
        chartData.labels.shift();
        chartData.datasets.forEach(dataset => { dataset.data.shift(); });
        }
        chartData.labels.push(newLabel);
        chartData.datasets[0].data.push(apiData.final_score);
        chartData.datasets[1].data.push(apiData.intermediate_scores.scoreA);
        chartData.datasets[2].data.push(apiData.intermediate_scores.scoreB);
        rebaChart.update();
    } catch(e) { console.error("Failed to update chart data:", e, apiData); }
}

/**
 * getUserMedia サポート確認
 */
function hasGetUserMedia() { return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia); }

/**
 * Webカメラを有効化 (環境カメラを要求)
 */
function enableCam() {
  if (!hasGetUserMedia()) { /* ... エラー処理 ... */ return; }
  if (!poseLandmarker) { /* ... エラー処理 ... */ return; }

  // ★★★ カメラ制約: 環境カメラ(背面)を再度指定 ★★★
  const constraints = {
    video: {
      facingMode: "environment"
      // width: { ideal: 1280 }, // 必要なら解像度指定
      // height: { ideal: 720 }
    }
  };
  console.log("Requesting camera with constraints:", constraints);

  navigator.mediaDevices.getUserMedia(constraints)
    .then((stream) => {
      video.srcObject = stream;
      const track = stream.getVideoTracks()[0];
      if (track) { console.log("Actual camera settings:", track.getSettings()); }

      video.addEventListener("loadeddata", () => {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
        lastVideoTime = -1;
        if (webcamRunning) { requestAnimationFrame(predictWebcam); }
      }, { once: true });
    })
    .catch((err) => { // エラーハンドリング (内容は変更なし、facingModeのエラーが出る可能性あり)
        console.error("Error accessing webcam:", err);
        let userErrorMessage = `Webカメラアクセスエラー (${err.name}: ${err.message})`;
        if (err.name === 'OverconstrainedError') { userErrorMessage = `要求されたカメラ設定(環境カメラ等)がサポートされていません。(${err.message})`; }
        else if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') { userErrorMessage = `カメラへのアクセスが許可されませんでした。設定を確認してください。`; }
        else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') { userErrorMessage = `利用可能なカメラが見つかりませんでした。`; }
        if (scoreDisplay) scoreDisplay.innerHTML = `<p style="color: red;">${userErrorMessage}</p>`;
        webcamRunning = false; webcamButton.innerText = "Recording Start";
     });
}

// ボタンイベントリスナー (変更なし)
webcamButton.addEventListener("click", () => {
  webcamRunning = !webcamRunning;
  webcamButton.innerText = webcamRunning ? "Stop Recording" : "Recording Start";
  if (webcamRunning) {
    if(scoreDisplay) scoreDisplay.innerHTML = "カメラを起動中...";
    enableCam();
  } else {
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(track => track.stop());
      video.srcObject = null; console.log("Webcam stream stopped.");
    }
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (scoreDisplay) scoreDisplay.innerHTML = "評価停止中";
  }
});

// キャリブレーション入力取得 (変更なし)
function getCalibrationInputs() { /* ... */
    return {
        filmingSide: document.querySelector('input[name="filmingSide"]:checked').value,
        neckRotation: Number(document.querySelector('input[name="neckRotation"]:checked').value),
        neckLateralBending: Number(document.querySelector('input[name="neckLateralBending"]:checked').value),
        trunkLateralFlexion: Number(document.querySelector('input[name="trunkLateralFlexion"]:checked').value),
        loadForce: Number(document.querySelector('input[name="loadForce"]:checked').value),
        postureCategory: document.querySelector('input[name="postureCategory"]:checked').value,
        upperArmCorrection: Number(document.querySelector('input[name="upperArmCorrection"]:checked').value),
        shoulderElevation: Number(document.querySelector('input[name="shoulderElevation"]:checked').value),
        gravityAssist: Number(document.querySelector('input[name="gravityAssist"]:checked').value),
        wristCorrection: Number(document.querySelector('input[name="wristCorrection"]:checked').value),
        staticPosture: Number(document.querySelector('input[name="staticPosture"]:checked').value),
        repetitiveMovement: Number(document.querySelector('input[name="repetitiveMovement"]:checked').value),
        unstableMovement: Number(document.querySelector('input[name="unstableMovement"]:checked').value),
        coupling: Number(document.querySelector('input[name="coupling"]:checked').value)
    };
}

/**
 * メインループ: 姿勢推定、描画、API呼び出し
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
        drawingUtils.drawLandmarks(landmarkSet, { radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1) });
        drawingUtils.drawConnectors(landmarkSet, PoseLandmarker.POSE_CONNECTIONS);

        // API スロットリング
        const now = performance.now();
        if (now - lastApiCallTime > apiCallInterval) {
          lastApiCallTime = now;
          const calibInputs = getCalibrationInputs();
          const payload = { landmarks: landmarkSet, calibInputs: calibInputs };

          // ★★★ API URL: 提供されたバックエンドURLに修正 ★★★
          const apiUrl = "https://reba-cgph.onrender.com/compute_reba";

          console.log("Calling API:", apiUrl);

          // API呼び出し (fetchとエラーハンドリングは変更なし)
          fetch(apiUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          })
          .then(response => { /* ... (エラーチェック含むレスポンス処理) ... */
              console.log("API response status:", response.status);
              if (!response.ok) {
                  return response.text().then(text => {
                     console.error("API error response body:", text);
                     try { const errData = JSON.parse(text); throw new Error(errData.detail || `HTTP error! Status: ${response.status}`); }
                     catch (e) { throw new Error(`HTTP error! Status: ${response.status}. Response: ${text}`); }
                  });
              }
              return response.json();
          })
          .then(data => { /* ... (スコア表示更新 + グラフ更新呼び出し) ... */
             console.log("API success response data:", data);
             if (scoreDisplay && webcamRunning) {
               scoreDisplay.innerHTML = `<p>最終REBAスコア: ${data.final_score}</p><p>リスクレベル: ${data.risk_level}</p>`;
             }
             if (webcamRunning) { updateChart(data); } // グラフ更新
          })
          .catch(err => { /* ... (エラー表示処理) ... */
            console.error("Full error object caught during API call:", err);
            let displayMessage = err.message || "不明なエラー";
            if (displayMessage.toLowerCase().includes('load failed') || displayMessage.toLowerCase().includes('failed to fetch')) {
                displayMessage = "APIへの接続または通信に失敗しました。URLとサーバーログを確認してください。";
            }
            if (scoreDisplay && webcamRunning) {
              scoreDisplay.innerHTML = `<p style="color: red;">エラー: REBAスコア取得失敗 (${displayMessage})</p>`;
            }
          });
        } // --- スロットリング終了 ---
      } // --- ランドマーク処理終了 ---
    }); // --- detectForVideo コールバック終了 ---
  } // --- video.readyState チェック終了 ---

  // 次のフレームを要求
  if (webcamRunning) { window.requestAnimationFrame(predictWebcam); }
}

// アプリケーション初期化
initPoseLandmarker();
