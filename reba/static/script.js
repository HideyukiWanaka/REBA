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
// static/script.js 内の getCalibrationInputs 関数全体を修正

// static/script.js 内

function getCalibrationInputs() {
  // HTMLのname属性のリスト
  const names_in_html = [
    "filmingSide", "neckRotation", "neckLateralBending", "trunkLateralFlexion",
    "loadForce", "shockForce", "postureCategory", "supportingLeg",
    "upperArmCorrection", "shoulderElevation", "gravityAssist", "wristCorrection",
    "wristAngleScore", // ← HTMLでのname属性
    "staticPosture", "repetitiveMovement", "unstableMovement", "coupling"
  ];
  // HTMLのnameとPythonモデルのフィールド名のマッピング (異なる場合のみ記述)
  const name_map = {
      "wristAngleScore": "wristBaseScore" // HTMLの 'wristAngleScore' を Pythonの 'wristBaseScore' にマップ
  };

  const data = {}; // Pythonバックエンドに送るデータオブジェクト
  let errorOccurred = false;

  for (const html_name of names_in_html) {
    // HTMLのname属性で要素を検索
    const element = document.querySelector(`input[name="${html_name}"]:checked`);
    // dataオブジェクトに格納する際のキー名 (Pythonモデルのフィールド名) を決定
    const python_name = name_map[html_name] || html_name;

    if (element) {
      // 値を取得し、適切な型で data オブジェクトに格納 (キーはpython_name)
      if (html_name === "filmingSide" || html_name === "postureCategory" || html_name === "supportingLeg") {
        data[python_name] = element.value;
      } else {
        const value = Number(element.value);
        data[python_name] = isNaN(value) ? 0 : value;
      }
    } else {
      console.warn(`Could not find checked input for name="${html_name}". Assigning default.`);
      errorOccurred = true;
      // デフォルト値を設定 (キーはpython_name)
      if (python_name === "filmingSide" || python_name === "postureCategory") { data[python_name] = ""; }
      else if (python_name === "supportingLeg") { data[python_name] = null; } // Optionalはnull
      else if (python_name === "wristBaseScore") { data[python_name] = 1; }   // wristBaseScoreのデフォルト
      else { data[python_name] = 0; }
    }
  }

  if (errorOccurred) {
      console.warn("Errors occurred fetching some calibration inputs...", data);
  }

  // ★★★ 最終チェック: Pythonの必須フィールドがdataに含まれるか ★★★
  // (supportingLegはOptionalなので除く)
  const requiredPythonKeys = [
      "filmingSide", "neckRotation", "neckLateralBending", "trunkLateralFlexion", "loadForce",
      "shockForce", "postureCategory", "upperArmCorrection", "shoulderElevation",
      "gravityAssist", "wristCorrection", "wristBaseScore", "staticPosture",
      "repetitiveMovement", "unstableMovement", "coupling"
  ];
  let missingKey = false;
  for (const key of requiredPythonKeys) {
      if (!(key in data)) {
          console.error(`CRITICAL: Required key "${key}" is missing from constructed calibInputs!`);
          missingKey = true;
      }
  }
  if (missingKey) return undefined; // 必須キーが欠けていれば未定義を返す (422の原因になりうる)
  // ★★★ ここまで最終チェック ★★★


  console.log("Successfully obtained calibration inputs:", data);
  return data; // 完成したdataオブジェクトを返す
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
        // predictWebcam 関数内
        // API スロットリング
        const now = performance.now();
        if (now - lastApiCallTime > apiCallInterval) {
          lastApiCallTime = now; // 最終呼び出し時間を更新

          // --- ★★★ 正しい処理フロー ★★★ ---
          const calibInputs = getCalibrationInputs(); // 1. 入力取得
          console.log("[DEBUG] Calibration Inputs:", calibInputs);

          const landmarkSet = result.landmarks[0]; // 2. ランドマーク取得

          // 3. 入力データチェック
          if (!landmarkSet || landmarkSet.length === 0 || typeof calibInputs !== 'object' || calibInputs === null) {
              console.error("Skipping API call due to invalid landmarks or calibInputs.", { landmarks: !!landmarkSet, calibInputs });
              if(scoreDisplay && webcamRunning) { scoreDisplay.innerHTML = "<p style='color:red;'>エラー: 入力/姿勢データ不備</p>"; }
              return; // データが不正なら中断
          }

          // 4. ペイロード作成
          const payload = {
              landmarks: landmarkSet,
              calibInputs: calibInputs
          };
          console.log("[DEBUG] Payload Object:", payload);

          // 5. JSON 文字列化 (同期エラーの可能性は低いが念のため try...catch)
          let jsonPayload;
          try {
              jsonPayload = JSON.stringify(payload);
              console.log("[DEBUG] Stringified Payload:", jsonPayload);
          } catch (stringifyError) {
              console.error("Error stringifying payload:", stringifyError, payload);
              if (scoreDisplay && webcamRunning) { scoreDisplay.innerHTML = `<p style="color: red;">エラー: 送信データ作成失敗</p>`; }
              return; // stringify 失敗時も中断
          }

          // 6. API 呼び出しと Promise 処理
          const apiUrl = "https://reba-cgph.onrender.com/compute_reba";
          console.log("Calling API:", apiUrl);

          fetch(apiUrl, { method: "POST", headers: { "Content-Type": "application/json" }, body: jsonPayload })
          .then(response => { // ① HTTP応答処理
              console.log("[DEBUG] API response status:", response.status);
              if (!response.ok) {
                  return response.text().then(text => {
                     console.error("[DEBUG] API error response body text:", text);
                     let errorMsg = `サーバーエラー Status: ${response.status}.`;
                     try { /* ... エラー詳細パース ... */ } catch (e) { /* ... */ }
                     throw new Error(errorMsg);
                  });
              }
              return response.json();
          })
          .then(data => { // ② 正常応答処理
             console.log("[DEBUG] API success data object:", data);
             if (!data) { throw new Error("API returned OK but data is null/undefined."); }
             // 最大スコア更新
             if (data?.final_score > maxRebaScore) maxRebaScore = data.final_score;
             // スコア表示更新
             if (scoreDisplay && webcamRunning) {
                 const score = data?.final_score ?? 'N/A';
                 const risk = data?.risk_level ?? 'N/A';
                 scoreDisplay.innerHTML = `<p>最終REBAスコア: ${score}</p><p>リスクレベル: ${risk}</p>`;
             }
             // グラフ更新
             if (webcamRunning) { updateChart(data); }
          })
          .catch(err => { // ③ エラー処理 (ネットワークエラーや throw されたエラー)
            console.error("[DEBUG] Error caught in fetch chain:", err);
            let displayMessage = err.message || "不明なAPIエラー";
            if (err.name === 'TypeError') { displayMessage = "API接続失敗"; }
            if (scoreDisplay && webcamRunning) { scoreDisplay.innerHTML = `<p style="color: red;">エラー: REBAスコア取得失敗 (${displayMessage})</p>`; }
          });
          // --- ★★★ ここまでが fetch 処理 ★★★ ---

        } // --- スロットリング if ブロック終了 ---
             // ★★★ Promise 用 .catch() はここまで ★★★

          } catch (stringifyError) { // ← 同期エラー(主にstringify)用の catch
              console.error("Error stringifying payload:", stringifyError, payload);
              if (scoreDisplay && webcamRunning) {
                  scoreDisplay.innerHTML = `<p style="color: red;">エラー: 送信データ作成失敗</p>`;
              }
          } // 同期エラー用 try-catch 終了

        } // --- スロットリング終了 ---
      } // --- ランドマーク処理終了 ---
    }); // --- detectForVideo コールバック終了 ---
  } // --- video.readyState チェック終了 ---
  if (webcamRunning) { window.requestAnimationFrame(predictWebcam); }
}

// アプリケーション初期化
initPoseLandmarker();
