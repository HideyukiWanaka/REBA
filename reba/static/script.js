// static/script.js

import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

// --- グローバル変数 ---
let poseLandmarker;
let runningMode = "VIDEO";
let webcamRunning = false;
let lastApiCallTime = 0;
const apiCallInterval = 500; // ms
let lastVideoTime = -1;
let maxRebaScore = 0; // ★ セッション中の最大REBAスコアを記録する変数 ★

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
const chartData = { /* ... */ }; // (内容は前回のまま)

/**
 * MediaPipe PoseLandmarkerを初期化 (Fullモデルを使用)
 */
async function initPoseLandmarker() {
    // ... (内容は変更なし: モデル初期化、グラフ初期化呼び出し等) ...
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
            baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task", delegate: "GPU" },
            runningMode: "VIDEO", numPoses: 1,
        });
        console.log("PoseLandmarker created successfully.");
        initChart(); // グラフ初期化
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
 * グラフを初期化 (変更なし)
 */
function initChart() { /* ... */ }

/**
 * グラフを更新 (変更なし)
 */
function updateChart(apiData) { /* ... */ }

/**
 * getUserMedia サポート確認 (変更なし)
 */
function hasGetUserMedia() { return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia); }

/**
 * Webカメラを有効化 (環境カメラ要求 - 変更なし)
 */
function enableCam() { /* ... (内容は変更なし) ... */
    if (!hasGetUserMedia()) { /*...*/ return; }
    if (!poseLandmarker) { /*...*/ return; }
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
 * ★ REBAスコアからリスクレベル文字列を取得するヘルパー関数 ★
 * @param {number | null} score REBAスコア
 * @returns {string} リスクレベル文字列
 */
function getRiskLevelText(score) {
  if (score === null || score === undefined || score <= 0) return "N/A"; // 0以下は無効とする
  if (score === 1) return "無視できる (Negligible)";
  if (score <= 3) return "低リスク (Low)";        // 2-3
  if (score <= 7) return "中リスク (Medium)";     // 4-7
  if (score <= 10) return "高リスク (High)";       // 8-10
  return "非常に高リスク (Very High)"; // 11-15
}

// --- ★ 開始/停止ボタンのイベントリスナー (修正) ★ ---
webcamButton.addEventListener("click", () => {
  webcamRunning = !webcamRunning; // 状態をトグル
  webcamButton.innerText = webcamRunning ? "Stop Recording" : "Recording Start";

  if (webcamRunning) {
    // --- 開始時 ---
    maxRebaScore = 0; // ★ 最大スコアをリセット ★
    // 結果表示欄を初期化
    if (scoreDisplay) scoreDisplay.innerHTML = "カメラを起動中...";
    // グラフデータをリセット (オプション)
    /* chartData.labels = [];
    chartData.datasets.forEach(dataset => { dataset.data = []; });
    if (rebaChart) rebaChart.update(); // グラフをクリア*/

    enableCam(); // カメラ有効化と予測ループ開始
  } else {
    // --- 停止時 ---
    // カメラストリームを停止
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(track => track.stop());
      video.srcObject = null;
      console.log("Webcam stream stopped.");
    }
    // Canvasをクリア
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // ★ 最大スコアとリスクレベルを表示 ★
    const maxRiskLevel = getRiskLevelText(maxRebaScore); // ヘルパー関数でリスクレベル取得
    if (scoreDisplay) {
      if (maxRebaScore > 0) { // 有効なスコアが記録されていれば
        scoreDisplay.innerHTML =
          `<h3>評価終了</h3>
           <p>今回の最大REBAスコア: <strong style="font-size: 1.2em;">${maxRebaScore}</strong></p>
           <p>対応リスクレベル: <strong style="font-size: 1.1em;">${maxRiskLevel}</strong></p>`;
      } else { // スコアが記録されなかった場合
        scoreDisplay.innerHTML = "評価停止中 (有効なスコアなし)";
      }
    }
    console.log(`Session stopped. Max REBA score was: ${maxRebaScore}`);
    // predictWebcamループは webcamRunning フラグにより自動停止する
  }
});

// キャリブレーション入力取得 (変更なし)
function getCalibrationInputs() { /* ... */ }

/**
 * メインループ: 姿勢推定、描画、API呼び出し (最大スコア更新処理を追加)
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
          const apiUrl = "https://reba-cgph.onrender.com/compute_reba"; // ★ ご自身のバックエンドURL ★

          console.log("Calling API:", apiUrl);

          // API呼び出し
          fetch(apiUrl, { /* ... */ })
          .then(response => { /* ... */ return response.json(); })
          .then(data => {
             console.log("API success response data:", data);

             // ★★★ 最大スコアを更新 ★★★
             if (data.final_score > maxRebaScore) {
               maxRebaScore = data.final_score;
               console.log(`New max REBA score recorded: ${maxRebaScore}`);
             }
             // ★★★ ここまで ★★★

             // 現在のスコア表示更新
             if (scoreDisplay && webcamRunning) {
               scoreDisplay.innerHTML =
                 `<p>最終REBAスコア: ${data.final_score}</p>
                  <p>リスクレベル: ${data.risk_level}</p>`;
             }
             // グラフ更新
             if (webcamRunning) { updateChart(data); }
          })
          .catch(err => { /* ... エラーハンドリング ... */ });
        } // --- スロットリング終了 ---
      } // --- ランドマーク処理終了 ---
    }); // --- detectForVideo コールバック終了 ---
  } // --- video.readyState チェック終了 ---

  // 次のフレームを要求
  if (webcamRunning) { window.requestAnimationFrame(predictWebcam); }
}

// アプリケーション初期化
initPoseLandmarker();

