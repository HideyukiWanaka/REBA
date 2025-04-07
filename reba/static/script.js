import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

let poseLandmarker;
let runningMode = "VIDEO"; // (4) "VIDEO" で初期化
let webcamRunning = false;
let lastApiCallTime = 0; // (3) APIスロットリング用
const apiCallInterval = 500; // (3) API呼び出し間隔 (ms)
let lastVideoTime = -1; // predictWebcam の重複実行防止用

// video と canvas 要素の取得
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);
const scoreDisplay = document.getElementById("scoreDisplay");
const webcamButton = document.getElementById("webcamButton");

// MediaPipe Pose Landmarker の初期化
async function initPoseLandmarker() {
  try {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        delegate: "GPU"
      },
      runningMode, // "VIDEO" で初期化された値を使用
      numPoses: 1,
    });
    console.log("PoseLandmarker initialized successfully.");
    // ボタンを有効化（任意）
    webcamButton.disabled = false;
  } catch (error) {
    console.error("Failed to initialize PoseLandmarker:", error);
    if (scoreDisplay) {
        scoreDisplay.innerHTML =
        `<p style="color: red;">エラー: モデルの初期化に失敗 (${error.message})</p>`;
    }
     webcamButton.disabled = true; // 初期化失敗時はボタンを無効化
  }
}

// getUserMedia サポート確認
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

// Webカメラ有効化関数
// Webカメラ有効化関数
function enableCam() {
  if (!hasGetUserMedia()) {
    console.warn("getUserMedia() is not supported by your browser");
    if (scoreDisplay) scoreDisplay.innerHTML = `<p style="color: red;">エラー: ブラウザがカメラ機能をサポートしていません</p>`;
    webcamRunning = false; // 実行フラグを戻す
    webcamButton.innerText = "Recording Start";
    return;
  }
  if (!poseLandmarker) {
    console.log("PoseLandmarker model not loaded yet.");
     if (scoreDisplay) scoreDisplay.innerHTML = `<p>モデルをロード中です。少々お待ちください...</p>`;
     // 必要であればリトライ処理
    return;
  }

  // カメラ設定を変更
  const constraints = {
    video: {
      // facingMode に "environment" を指定して背面カメラを要求
      facingMode: "environment"
      // 必要であれば他の制約（解像度など）も追加できます
      // width: { ideal: 1280 },
      // height: { ideal: 720 }
    }
  };
  console.log("Requesting camera with constraints:", constraints); // 確認用ログ

  // カメラアクセス開始
  navigator.mediaDevices.getUserMedia(constraints)
    .then((stream) => {
      video.srcObject = stream;

      // (オプション) 実際に取得したカメラ設定を確認
      const track = stream.getVideoTracks()[0];
      const settings = track.getSettings();
      console.log("Actual camera settings obtained:", settings);
      if (settings.facingMode) {
          console.log(`Using camera facing: ${settings.facingMode}`);
      }

      video.addEventListener("loadeddata", () => {
        // (1) Canvas解像度をビデオの実際のサイズに合わせる
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
        // ループ開始
        lastVideoTime = -1; // リセット
        requestAnimationFrame(predictWebcam);
      });
    })
    .catch((err) => {
      console.error("Error accessing webcam:", err);
      // エラーメッセージに制約に関する情報を含める（例）
      let userErrorMessage = `Webカメラにアクセスできません (${err.message})`;
      if (err.name === 'OverconstrainedError') {
          userErrorMessage = `指定されたカメラ（環境カメラなど）が見つからないか、設定がサポートされていません。(${err.message})`;
      } else if (err.name === 'NotAllowedError') {
           userErrorMessage = `カメラへのアクセスが許可されませんでした。ブラウザの設定を確認してください。`;
      }
      if (scoreDisplay) {
          scoreDisplay.innerHTML = `<p style="color: red;">エラー: ${userErrorMessage}</p>`;
      }
      webcamRunning = false; // 実行フラグを戻す
      webcamButton.innerText = "Recording Start";
    });
}


  // カメラ設定（特定の解像度を要求しない方が互換性が高い場合がある）
  const constraints = { video: true };

  // カメラアクセス開始
  navigator.mediaDevices.getUserMedia(constraints)
    .then((stream) => {
      video.srcObject = stream;
      video.addEventListener("loadeddata", () => {
        // (1) Canvas解像度をビデオの実際のサイズに合わせる
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
        // ループ開始
        lastVideoTime = -1; // リセット
        requestAnimationFrame(predictWebcam);
      });
    })
    .catch((err) => {
      console.error("Error accessing webcam:", err);
      if (scoreDisplay) {
          scoreDisplay.innerHTML =
          `<p style="color: red;">エラー: Webカメラにアクセスできません (${err.message})</p>`;
      }
      webcamRunning = false; // 実行フラグを戻す
      webcamButton.innerText = "Recording Start";
    });
}

// ボタンのイベントリスナー
webcamButton.addEventListener("click", () => {
  if (!poseLandmarker) {
    console.log("PoseLandmarker model not loaded yet.");
    if (scoreDisplay) scoreDisplay.innerHTML = `<p>モデルをロード中です。少々お待ちください...</p>`;
    return;
  }

  webcamRunning = !webcamRunning; // フラグをトグル
  webcamButton.innerText = webcamRunning ? "Stop Recording" : "Recording Start";

  if (webcamRunning) {
    if(scoreDisplay) scoreDisplay.innerHTML = "評価開始...";
    enableCam(); // カメラと予測を開始
  } else {
    // (2) カメラストリーム停止処理
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(track => track.stop());
      video.srcObject = null;
      console.log("Webcam stream stopped.");
    }
    // ループは requestAnimationFrame の条件で自動停止
    // 描画クリアと表示リセット
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (scoreDisplay) scoreDisplay.innerHTML = "評価停止中";
  }
});

// ===== 補正入力の取得 =====
function getCalibrationInputs() {
  // この関数は変更なし
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
    coupling: Number(document.querySelector('input[name="coupling"]:checked').value) // 0, 1, 2, or 3
  };
}

// ===== Webカメラからの検出ループ =====
async function predictWebcam() {
  // webcamRunning フラグが false ならループを即時終了
  if (!webcamRunning) {
    return;
  }

  // ビデオの準備ができているか、新しいフレームかを確認
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const startTimeMs = performance.now();

    // MediaPipe 検出実行
    poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
      // webcamRunning が false になっていたら描画やAPI呼び出しをスキップ
      if (!webcamRunning) return;

      // Canvasクリア
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

      // ランドマークがあれば描画とAPI呼び出し
      if (result.landmarks && result.landmarks.length > 0) {
        const landmarkSet = result.landmarks[0]; // numPoses: 1

        // 描画
        drawingUtils.drawLandmarks(landmarkSet, {
          radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
        });
        drawingUtils.drawConnectors(landmarkSet, PoseLandmarker.POSE_CONNECTIONS);

        // (3) API呼び出しスロットリング
        const now = performance.now();
        if (now - lastApiCallTime > apiCallInterval) {
          lastApiCallTime = now; // 最終呼び出し時間を更新

          const calibInputs = getCalibrationInputs();
          const payload = {
            landmarks: landmarkSet,
            calibInputs: calibInputs
          };

          // !! 重要: デプロイ後にバックエンドURLに書き換えてください !!
          const apiUrl = "https://reba-cgph.onrender.com"; // ローカル用
          // const apiUrl = "https://your-backend-name.onrender.com/compute_reba"; // Renderデプロイ後のURL例

          // バックエンド API へ送信
          fetch(apiUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          })
          .then(response => {
              if (!response.ok) { // エラーレスポンスチェック
                  // エラー情報をJSONとして取得試行
                  return response.json().then(errData => {
                     // FastAPIのHTTPExceptionはdetailにメッセージが入ることが多い
                     throw new Error(errData.detail || `HTTPエラー! status: ${response.status}`);
                  }).catch(() => {
                      // JSON取得失敗時はステータスコードでエラー
                      throw new Error(`HTTPエラー! status: ${response.status}`);
                  });
              }
              return response.json(); // 正常時はJSONをパース
          })
          .then(data => {
            // API呼び出し成功時の処理
            if (scoreDisplay && webcamRunning) { // webcamRunning を再確認
              scoreDisplay.innerHTML =
                `<p>最終REBAスコア: ${data.final_score}</p>
                 <p>リスクレベル: ${data.risk_level}</p>`;
            }
          })
          .catch(err => { // (5) エラーハンドリング強化
            console.error("Error calling REBA API:", err);
            if (scoreDisplay && webcamRunning) { // webcamRunning を再確認
              scoreDisplay.innerHTML =
                `<p style="color: red;">エラー: REBAスコア取得失敗 (${err.message})</p>`;
            }
          });
        } // --- End throttling check ---
      } // --- End if(result.landmarks) ---
    }); // --- End detectForVideo callback ---
  } // --- End if(video.readyState) ---

  // webcamRunning フラグが true の間、次のフレームを要求
  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}

// 初期化を実行
initPoseLandmarker();
// 初期状態ではボタンを無効化しておき、初期化完了後に有効化
webcamButton.disabled = true;

