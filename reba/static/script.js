// static/script.js

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
let rebaChart = null; // Chart.js インスタンス
const chartDataPoints = 60; // グラフに表示する最大データ点数
const chartData = {
  labels: [], // X軸ラベル
  datasets: [
    {
      label: 'REBA Total',
      data: [],
      borderColor: 'rgb(255, 99, 132)', // Red
      backgroundColor: 'rgba(255, 99, 132, 0.1)', // Slightly transparent fill
      tension: 0.1,
      pointRadius: 0 // Hide points for smoother line
    },
    {
      label: 'Score A',
      data: [],
      borderColor: 'rgb(54, 162, 235)', // Blue
      backgroundColor: 'rgba(54, 162, 235, 0.1)',
      tension: 0.1,
      pointRadius: 0
    },
    {
      label: 'Score B',
      data: [],
      borderColor: 'rgb(75, 192, 192)', // Green
      backgroundColor: 'rgba(75, 192, 192, 0.1)',
      tension: 0.1,
      pointRadius: 0
    }
  ]
};
// --- ★ グラフ用変数ここまで ★ ---

/**
 * MediaPipe PoseLandmarkerを非同期で初期化
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

    if (scoreDisplay) scoreDisplay.innerHTML = "<p>姿勢推定モデル(full)をダウンロード中...</p>";
    console.log("Resolver fetched. Creating PoseLandmarker (full)...");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task", // Liteモデル
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
    console.log("PoseLandmarker created successfully.");
    initChart(); // ★ PoseLandmarker成功後にグラフを初期化 ★
    webcamButton.disabled = false;
    webcamButton.innerText = "Recording Start";
    if (scoreDisplay) scoreDisplay.innerHTML = "モデル準備完了。ボタンを押して開始してください。";
  } catch (error) {
    console.error("Failed to initialize PoseLandmarker:", error);
    webcamButton.disabled = true;
    webcamButton.innerText = "Load Failed";
    let errorMsg = `モデル初期化失敗: ${error.message}`;
    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError') || error instanceof TypeError) {
        errorMsg = "モデルのダウンロードに失敗しました。ネットワーク接続を確認するか、安定したWi-Fi環境で再試行してください。";
    } else if (error.message.includes('Wasm') || error.message.includes('WebGL')) {
        errorMsg = "ブラウザまたはデバイスがモデル実行に必要な機能をサポートしていない可能性があります。";
    }
    if (scoreDisplay) {
        scoreDisplay.innerHTML = `<p style="color: red;">エラー: ${errorMsg}</p>`;
    }
  }
}

/**
 * ★ Chart.js を使ってグラフを初期化する関数 ★
 */
function initChart() {
  try {
    const ctx = document.getElementById('rebaChart').getContext('2d');
    if (!ctx) {
        console.error("Chart canvas element 'rebaChart' not found.");
        return;
    }
    rebaChart = new Chart(ctx, {
      type: 'line',
      data: chartData,
      options: {
        responsive: true,
        maintainAspectRatio: false, // コンテナに合わせて伸縮させる場合はfalse推奨
        scales: {
          y: {
            beginAtZero: true,
            suggestedMax: 15, // REBA Max Score
            title: { display: true, text: 'Score' } // Y軸タイトル
          },
          x: {
             ticks: {
                 callback: function(value, index, values) {
                     // Display every Nth label to prevent clutter
                     const N = Math.ceil(chartDataPoints / 10); // Show ~10 labels max
                     return index % N === 0 ? this.getLabelForValue(value) : null;
                 },
                 autoSkip: false,
                 maxRotation: 0,
                 minRotation: 0
             },
             title: { display: true, text: 'Time (Sequence)' } // X軸タイトル
          }
        },
        animation: { duration: 0 }, // Disable animation for real-time
        plugins: {
            legend: { position: 'bottom' }, // 凡例を下部に表示
            title: { display: false }
        },
        // パフォーマンス向上のためのオプション (オプション)
        // parsing: false, // データ構造が正しい場合、解析をスキップ
        // normalized: true, // データが正規化されている場合
      }
    });
    console.log("Chart initialized successfully.");
  } catch(e) {
      console.error("Failed to initialize chart:", e);
      if(scoreDisplay) scoreDisplay.innerHTML += "<p style='color:red;'>グラフの初期化に失敗しました。</p>";
  }
}

/**
 * ★ APIから受け取ったデータでグラフを更新する関数 ★
 * @param {object} apiData - バックエンドAPIからのレスポンスデータ
 */
function updateChart(apiData) {
  if (!rebaChart || !apiData || !apiData.intermediate_scores) {
    return; // グラフ未初期化 or データ不足
  }

  try {
    // X軸ラベル (シーケンス番号 or 時刻)
    const newLabel = chartData.labels.length > 0 ? Number(chartData.labels[chartData.labels.length - 1]) + 1 : 1;
    // const newLabel = new Date().toLocaleTimeString(); // 時刻を使う場合

    // データ点数が最大値を超えていたら古いデータを削除
    while (chartData.labels.length >= chartDataPoints) {
      chartData.labels.shift();
      chartData.datasets.forEach(dataset => {
        dataset.data.shift();
      });
    }

    // 新しいデータを追加
    chartData.labels.push(newLabel);
    chartData.datasets[0].data.push(apiData.final_score);
    chartData.datasets[1].data.push(apiData.intermediate_scores.scoreA);
    chartData.datasets[2].data.push(apiData.intermediate_scores.scoreB);

    // グラフを更新
    rebaChart.update();
  } catch(e) {
      console.error("Failed to update chart data:", e, apiData);
  }
}


/**
 * ブラウザがカメラ機能(getUserMedia)をサポートしているか確認
 */
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

/**
 * Webカメラを有効化し、予測ループを開始
 */
function enableCam() {
  if (!hasGetUserMedia()) { /* ... (エラー処理) ... */ return; }
  if (!poseLandmarker) { /* ... (エラー処理) ... */ return; }

  // カメラ制約 (デフォルトカメラを使用)
  const constraints = { video: true };
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
    .catch((err) => { /* ... (エラー処理, メッセージ表示, フラグリセット) ... */
        console.error("Error accessing webcam:", err);
        let userErrorMessage = `Webカメラアクセスエラー (${err.name}: ${err.message})`;
        // エラーの種類に応じてメッセージ調整 (省略)
        if (scoreDisplay) scoreDisplay.innerHTML = `<p style="color: red;">${userErrorMessage}</p>`;
        webcamRunning = false;
        webcamButton.innerText = "Recording Start";
     });
}

// 開始/停止ボタンのイベントリスナー
webcamButton.addEventListener("click", () => {
  webcamRunning = !webcamRunning;
  webcamButton.innerText = webcamRunning ? "Stop Recording" : "Recording Start";
  if (webcamRunning) {
    if(scoreDisplay) scoreDisplay.innerHTML = "カメラを起動中...";
    enableCam();
  } else {
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(track => track.stop());
      video.srcObject = null;
      console.log("Webcam stream stopped.");
    }
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (scoreDisplay) scoreDisplay.innerHTML = "評価停止中";
    // グラフデータもクリアする (オプション)
    // chartData.labels = [];
    // chartData.datasets.forEach(dataset => { dataset.data = []; });
    // if (rebaChart) rebaChart.update();
  }
});

/**
 * HTMLフォームからキャリブレーション設定値を取得
 */
function getCalibrationInputs() {
  // 変更なし
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
 * Webカメラ映像から姿勢を推定し、結果を描画・API送信するメインループ関数
 */
async function predictWebcam() {
  if (!webcamRunning) return; // ループ停止

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const startTimeMs = performance.now();

    poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
      if (!webcamRunning) return; // コールバック中に停止した場合

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

          // !! 重要: デプロイ後にバックエンドURLに書き換えてください !!
          const apiUrl = "http://127.0.0.1:8000/compute_reba"; // ローカル開発用
          // const apiUrl = "https://your-backend-name.onrender.com/compute_reba"; // Renderデプロイ後のURL例

          console.log("Calling API:", apiUrl); // API呼び出しURL確認用

          fetch(apiUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          })
          .then(response => {
              console.log("API response status:", response.status);
              if (!response.ok) {
                  return response.text().then(text => {
                     console.error("API error response body:", text);
                     try {
                         const errData = JSON.parse(text);
                         throw new Error(errData.detail || `HTTP error! Status: ${response.status}`);
                     } catch (e) {
                         throw new Error(`HTTP error! Status: ${response.status}. Response: ${text}`);
                     }
                  });
              }
              return response.json();
          })
          .then(data => {
             console.log("API success response data:", data);
             // スコア表示更新
             if (scoreDisplay && webcamRunning) {
               scoreDisplay.innerHTML =
                 `<p>最終REBAスコア: ${data.final_score}</p>
                  <p>リスクレベル: ${data.risk_level}</p>`;
             }
             // ★ グラフ更新 ★
             if (webcamRunning) {
                 updateChart(data);
             }
          })
          .catch(err => {
            console.error("Full error object caught during API call:", err);
            let displayMessage = err.message || "不明なエラー";
            if (displayMessage.toLowerCase().includes('load failed') || displayMessage.toLowerCase().includes('failed to fetch')) {
                displayMessage = "APIへの接続または通信に失敗しました。URLとサーバーログを確認してください。";
            }
            if (scoreDisplay && webcamRunning) {
              scoreDisplay.innerHTML =
                `<p style="color: red;">エラー: REBAスコア取得失敗 (${displayMessage})</p>`;
            }
          });
        } // --- スロットリング終了 ---
      } // --- ランドマーク処理終了 ---
    }); // --- detectForVideo コールバック終了 ---
  } // --- video.readyState チェック終了 ---

  // 次のフレームを要求
  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}

// アプリケーション初期化
initPoseLandmarker();

