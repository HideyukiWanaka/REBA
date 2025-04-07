// static/script.js

import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

let poseLandmarker;
let runningMode = "VIDEO"; // "VIDEO" で初期化
let webcamRunning = false;
let lastApiCallTime = 0; // APIスロットリング用
const apiCallInterval = 500; // API呼び出し間隔 (ms)
let lastVideoTime = -1; // predictWebcam の重複実行防止用

// DOM要素を取得
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);
const scoreDisplay = document.getElementById("scoreDisplay");
const webcamButton = document.getElementById("webcamButton");

/**
 * MediaPipe PoseLandmarkerを非同期で初期化する関数
 */
async function initPoseLandmarker() {
  // UIをロード中状態に更新
  if (scoreDisplay) scoreDisplay.innerHTML = "<p>姿勢推定モデルの準備を開始...</p>";
  webcamButton.disabled = true; // ボタンを無効化
  webcamButton.innerText = "Loading..."; // ボタンテキスト変更

  console.log("Initializing PoseLandmarker...");
  try {
    if (scoreDisplay) scoreDisplay.innerHTML = "<p>実行ファイルをダウンロード中...</p>";
    console.log("Fetching vision tasks resolver...");
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");

    if (scoreDisplay) scoreDisplay.innerHTML = "<p>姿勢推定モデル(lite)をダウンロード中...</p>"; // Liteモデル使用を明記
    console.log("Resolver fetched. Creating PoseLandmarker (lite)...");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        // Liteモデルを使用 (ネットワーク負荷軽減のため)
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        delegate: "GPU" // デフォルトはGPU (問題があれば "CPU" も試す価値あり)
      },
      runningMode: "VIDEO",
      numPoses: 1, // 検出する人数を1人に制限
    });
    console.log("PoseLandmarker created successfully:", poseLandmarker);
    webcamButton.disabled = false; // 初期化成功したらボタンを有効化
    webcamButton.innerText = "Recording Start";
    if (scoreDisplay) scoreDisplay.innerHTML = "モデル準備完了。ボタンを押して開始してください。";
  } catch (error) {
    console.error("Failed to initialize PoseLandmarker:", error);
    webcamButton.disabled = true; // エラー時はボタンを無効のまま
    webcamButton.innerText = "Load Failed"; // エラー表示
    // より詳細なエラーメッセージをUIに表示
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
 * ブラウザがカメラ機能(getUserMedia)をサポートしているか確認
 * @returns {boolean} サポートしていれば true
 */
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

/**
 * Webカメラを有効化し、映像ストリームを取得してビデオ要素に設定、
 * その後、姿勢推定ループを開始する
 */
function enableCam() {
  // 機能サポートチェック
  if (!hasGetUserMedia()) {
    console.warn("getUserMedia() is not supported by your browser");
    if (scoreDisplay) scoreDisplay.innerHTML = `<p style="color: red;">エラー: ブラウザがカメラ機能をサポートしていません</p>`;
    webcamRunning = false; // 実行フラグをリセット
    webcamButton.innerText = "Recording Start";
    return;
  }
  // モデル初期化済みチェック
  if (!poseLandmarker) {
    console.log("PoseLandmarker model not loaded yet.");
    if (scoreDisplay) scoreDisplay.innerHTML = `<p>エラー: モデルがロードされていません。</p>`;
    webcamRunning = false; // 実行フラグをリセット
    webcamButton.innerText = "Recording Start";
    return;
  }

  // ★ カメラへのアクセス制約 (環境カメラを要求) ★
  const constraints = {
    video: {
      facingMode: "environment"
      // 必要であれば他の制約も追加
      // width: { ideal: 1280 },
      // height: { ideal: 720 }
    }
  };
  console.log("Requesting camera with constraints:", constraints); // 要求する制約をログに出力

  // getUserMediaでカメラアクセスを要求
  navigator.mediaDevices.getUserMedia(constraints)
    .then((stream) => {
      video.srcObject = stream; // 取得したストリームをvideo要素に接続

      // (任意)実際に取得したカメラ設定をログに出力
      const track = stream.getVideoTracks()[0];
      if (track) {
        const settings = track.getSettings();
        console.log("Actual camera settings obtained:", settings);
         // facingModeが実際にenvironmentになったか確認
         if (settings.facingMode) {
             console.log(`Using camera facing: ${settings.facingMode}`);
         } else {
             console.log("Facing mode could not be determined or is not 'environment'.");
         }
      }

      // ビデオのメタデータが読み込まれたら、Canvasサイズを設定し予測ループ開始
      video.addEventListener("loadeddata", () => {
        // Canvasの内部解像度をビデオの実際の解像度に合わせる
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
        // 予測ループのタイムスタンプをリセット
        lastVideoTime = -1;
        // webcamRunningフラグがtrueの場合のみループを開始
        if (webcamRunning) {
             requestAnimationFrame(predictWebcam);
        }
      }, { once: true }); // イベントリスナーを一度だけ実行するように設定
    })
    .catch((err) => { // カメラアクセス失敗時のエラーハンドリング
      console.error("Error accessing webcam:", err);
      let userErrorMessage = `Webカメラにアクセスできません (${err.message})`;
      // エラーの種類に応じてメッセージを具体化
      if (err.name === 'OverconstrainedError') {
          userErrorMessage = `要求されたカメラ設定（特に環境カメラ）がサポートされていないか、見つかりません。(${err.message})`; // メッセージを少し変更
      } else if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
           userErrorMessage = `カメラへのアクセスが許可されませんでした。ブラウザやOSの設定を確認してください。`;
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
           userErrorMessage = `利用可能なカメラが見つかりませんでした。`;
      }
      if (scoreDisplay) {
          scoreDisplay.innerHTML = `<p style="color: red;">エラー: ${userErrorMessage}</p>`;
      }
      webcamRunning = false; // 実行フラグをリセット
      webcamButton.innerText = "Recording Start";
    });
}

// 開始/停止ボタンのクリックイベントリスナー
webcamButton.addEventListener("click", () => {
  // 実行状態をトグル
  webcamRunning = !webcamRunning;
  webcamButton.innerText = webcamRunning ? "Stop Recording" : "Recording Start";

  if (webcamRunning) {
    // 開始処理
    if(scoreDisplay) scoreDisplay.innerHTML = "カメラを起動中...";
    enableCam(); // カメラ有効化と予測ループ開始処理を呼び出す
  } else {
    // 停止処理
    // Webカメラのストリームを停止
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(track => track.stop());
      video.srcObject = null; // video要素との接続を解除
      console.log("Webcam stream stopped.");
    }
    // Canvasをクリア
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    // スコア表示をリセット
    if (scoreDisplay) scoreDisplay.innerHTML = "評価停止中";
    // 次の predictWebcam 呼び出しを防ぐ (requestAnimationFrameの条件で停止)
  }
});

/**
 * HTMLフォームから現在のキャリブレーション設定値を取得する関数
 * @returns {object} キャリブレーション設定値を含むオブジェクト
 */
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

/**
 * Webカメラ映像から姿勢を推定し、結果を描画・API送信するメインループ関数
 */
async function predictWebcam() {
  // webcamRunning フラグが false ならループを即時終了
  if (!webcamRunning) {
    return;
  }

  // ビデオの準備ができているか、新しいフレームかを確認
  // video.readyState >= 2 は HAVE_CURRENT_DATA かそれ以上を示す
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime; // 現在のフレーム時間を記録
    const startTimeMs = performance.now(); // 検出処理の開始時間

    // MediaPipe Pose Landmarker で姿勢を検出
    poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
      // 非同期コールバック実行時にも webcamRunning フラグを再確認
      if (!webcamRunning) return;

      // 前回の描画をクリア
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

      // ランドマークが検出された場合のみ処理
      if (result.landmarks && result.landmarks.length > 0) {
        const landmarkSet = result.landmarks[0]; // 最初の人物のランドマーク

        // Canvasにランドマークと骨格を描画
        drawingUtils.drawLandmarks(landmarkSet, {
          radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1) // Z座標に応じて点を描画(奥は小さく)
        });
        drawingUtils.drawConnectors(landmarkSet, PoseLandmarker.POSE_CONNECTIONS); // 線で結ぶ

        // API呼び出し頻度を制限 (スロットリング)
        const now = performance.now();
        if (now - lastApiCallTime > apiCallInterval) {
          lastApiCallTime = now; // 最後にAPI呼び出しした時間を更新

          // キャリブレーション設定を取得
          const calibInputs = getCalibrationInputs();
          // APIに送信するデータを作成
          const payload = {
            landmarks: landmarkSet, // 検出されたランドマークデータ
            calibInputs: calibInputs // フォームからの設定値
          };

          // !! 重要: デプロイ後にバックエンドURLに書き換えてください !!
          const apiUrl = "https://your-backend-name.onrender.com/compute_reba"; // ローカル開発用
          // const apiUrl = "https://your-backend-name.onrender.com/compute_reba"; // Renderデプロイ後のURL例

          // バックエンドAPIにPOSTリクエストを送信
          fetch(apiUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          })
          .then(response => { // レスポンスのステータスコードをチェック
              if (!response.ok) {
                  // エラーレスポンスの場合は、内容を解析してエラーを投げる
                  return response.json().then(errData => {
                     // FastAPIからのエラー詳細(detail)があればそれを使う
                     throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
                  }).catch(() => {
                      // JSON解析失敗時はステータスコードのみ
                      throw new Error(`HTTP error! status: ${response.status}`);
                  });
              }
              return response.json(); // 正常ならJSONをパース
          })
          .then(data => { // APIからの正常なレスポンス受信時
            // スコア表示を更新（webcamRunningがtrueの場合のみ）
            if (scoreDisplay && webcamRunning) {
              scoreDisplay.innerHTML =
                `<p>最終REBAスコア: ${data.final_score}</p>
                 <p>リスクレベル: ${data.risk_level}</p>`;
            }
          })
          .catch(err => { // fetch自体またはレスポンス処理中のエラー
            console.error("Error calling REBA API:", err);
            // エラーをUIに表示（webcamRunningがtrueの場合のみ）
            if (scoreDisplay && webcamRunning) {
              scoreDisplay.innerHTML =
                `<p style="color: red;">エラー: REBAスコア取得失敗 (${err.message})</p>`;
            }
          });
        } // --- スロットリングブロック終了 ---
      } // --- ランドマーク処理終了 ---
    }); // --- detectForVideo コールバック終了 ---
  } // --- video.readyState チェック終了 ---

  // webcamRunning フラグが true の間、次のアニメーションフレームで再度実行を要求
  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}

// アプリケーション初期化関数を呼び出し
initPoseLandmarker();
