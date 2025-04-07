// static/script.js

import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

// --- グローバル変数 ---
let poseLandmarker;
let runningMode = "VIDEO";
let webcamRunning = false;
let lastApiCallTime = 0;
const apiCallInterval = 500; // ms
let lastVideoTime = -1;
let maxRebaScore = 0; // セッション中の最大REBAスコア

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
        // ★ Fullモデルのパスを使用 ★
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO", numPoses: 1,
    });
    console.log("PoseLandmarker created successfully.");
    initChart(); // ★ グラフ初期化をここに移動 ★
    webcamButton.disabled = false; webcamButton.innerText = "Recording Start";
    if (scoreDisplay) scoreDisplay.innerHTML = "モデル準備完了。ボタンを押して開始してください。";
  } catch (error) {
    // ... (エラーハンドリングは変更なし) ...
    console.error("Failed to initialize PoseLandmarker:", error);
    webcamButton.disabled = true; webcamButton.innerText = "Load Failed";
    let errorMsg = `モデル初期化失敗: ${error.message}`;
    // ... (エラーメッセージ詳細化) ...
    if (scoreDisplay) scoreDisplay.innerHTML = `<p style="color: red;">エラー: ${errorMsg}</p>`;
  }
}

/**
 * グラフを初期化
 */
function initChart() {
  // すでに初期化済みの場合は何もしない（安全策）
  if (rebaChart) { return; }
  try {
    const ctx = document.getElementById('rebaChart').getContext('2d');
    if (!ctx) { console.error("Chart canvas element 'rebaChart' not found."); return; }
    rebaChart = new Chart(ctx, {
      type: 'line', data: chartData,
      options: {
          responsive: true, maintainAspectRatio: false,
          scales: { /* ... 軸設定 ... */ }, animation: { duration: 0 }, plugins: { /* ... 凡例など ... */ }
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
  // グラフ未初期化 or データ不足 or intermediate_scoresがない場合は更新しない
  if (!rebaChart || !apiData || !apiData.intermediate_scores) {
     console.warn("Chart update skipped. Chart not ready or data missing intermediate_scores.", apiData);
     return;
  }
  try {
    const newLabel = chartData.labels.length > 0 ? Number(chartData.labels[chartData.labels.length - 1]) + 1 : 1;
    // データ点数制限
    while (chartData.labels.length >= chartDataPoints) {
      chartData.labels.shift();
      chartData.datasets.forEach(dataset => { dataset.data.shift(); });
    }
    // データ追加
    chartData.labels.push(newLabel);
    // ★ data.intermediate_scoresが存在するか確認してからアクセス ★
    chartData.datasets[0].data.push(apiData.final_score ?? null); // nullish coalescing で安全に
    chartData.datasets[1].data.push(apiData.intermediate_scores.scoreA ?? null);
    chartData.datasets[2].data.push(apiData.intermediate_scores.scoreB ?? null);

    rebaChart.update(); // グラフ再描画
  } catch(e) {
      console.error("Failed to update chart data:", e, apiData);
  }
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
      // ... (ログ、loadeddataリスナー) ...
      video.addEventListener("loadeddata", () => {
          canvasElement.width = video.videoWidth;
          canvasElement.height = video.videoHeight;
          lastVideoTime = -1;
          if (webcamRunning) { requestAnimationFrame(predictWebcam); }
      }, { once: true });
    })
    .catch((err) => { /* ... (エラーハンドリング) ... */ });
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

// --- ★ 開始/停止ボタンのイベントリスナー (修正) ★ ---
webcamButton.addEventListener("click", () => {
  webcamRunning = !webcamRunning;
  webcamButton.innerText = webcamRunning ? "Stop Recording" : "Recording Start";

  if (webcamRunning) {
    // --- 開始時 ---
    maxRebaScore = 0; // 最大スコアをリセット
    if(scoreDisplay) scoreDisplay.innerHTML = "カメラを起動中...";

    // ★ グラフデータを安全にリセット ★
    chartData.labels = []; // データ配列を空にする
    chartData.datasets.forEach(dataset => { dataset.data = []; });
    if (rebaChart) { // グラフオブジェクトが存在すればupdateを呼ぶ
        rebaChart.update();
    } else {
        // もしinitChartがまだ呼ばれていなくても、データは空になっている
        console.warn("Chart not initialized yet when trying to reset data.");
    }

    enableCam(); // カメラ有効化と予測ループ開始
  } else {
    // --- 停止時 ---
    if (video.srcObject) { /* ... カメラ停止 ... */ }
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // ★ 最大スコア表示 (変更なし) ★
    const maxRiskLevel = getRiskLevelText(maxRebaScore);
    if (scoreDisplay) {
      if (maxRebaScore > 0) {
        scoreDisplay.innerHTML = `<h3>評価終了</h3><p>今回の最大REBAスコア: <strong style="font-size: 1.2em;">${maxRebaScore}</strong></p><p>対応リスクレベル: <strong style="font-size: 1.1em;">${maxRiskLevel}</strong></p>`;
      } else {
        scoreDisplay.innerHTML = "評価停止中 (有効なスコアなし)";
      }
    }
    console.log(`Session stopped. Max REBA score was: ${maxRebaScore}`);
  }
});

// キャリブレーション入力取得 (変更なし)
function getCalibrationInputs() { /* ... */ }

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
          // ★★★ 必ずデプロイしたバックエンドURLに書き換えてください ★★★
          const apiUrl = "https://reba-cgph.onrender.com/compute_reba";
          console.log("Calling API:", apiUrl);

          try { // fetchを含むtry-catchを追加(stringifyエラー等の捕捉)
              const jsonPayload = JSON.stringify(payload);
              fetch(apiUrl, { method: "POST", headers: { "Content-Type": "application/json" }, body: jsonPayload })
              .then(response => { /* ... */ return response.json(); })
              .then(data => {
                 console.log("API success data:", data);
                 // ★ 最大スコア更新 ★
                 if (data && data.final_score !== null && data.final_score !== undefined && data.final_score > maxRebaScore) {
                   maxRebaScore = data.final_score;
                   console.log(`New max REBA score recorded: ${maxRebaScore}`);
                 }
                 // 現在スコア表示
                 if (scoreDisplay && webcamRunning) {
                   // dataが存在し、必要なキーがあるか確認(より安全に)
                   const score = data?.final_score ?? 'N/A';
                   const risk = data?.risk_level ?? 'N/A';
                   scoreDisplay.innerHTML = `<p>最終REBAスコア: ${score}</p><p>リスクレベル: ${risk}</p>`;
                 }
                 // グラフ更新
                 if (webcamRunning) { updateChart(data); }
              })
              .catch(err => { /* ... エラーハンドリング ... */ });
          } catch (stringifyError) {
              console.error("Error stringifying payload:", stringifyError, payload);
              if (scoreDisplay && webcamRunning) { scoreDisplay.innerHTML = `<p style="color: red;">エラー: 送信データの作成失敗</p>`; }
          }
        } // --- スロットリング終了 ---
      } // --- ランドマーク処理終了 ---
    }); // --- detectForVideo コールバック終了 ---
  } // --- video.readyState チェック終了 ---

  // 次のフレームを要求
  if (webcamRunning) { window.requestAnimationFrame(predictWebcam); }
}

// アプリケーション初期化
initPoseLandmarker();

