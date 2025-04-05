import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");
let poseLandmarker = undefined;
let runningMode = "IMAGE";
let webcamRunning = false;
const videoHeight = "360px";
const videoWidth = "480px";

// MediaPipe Pose Landmarker の初期化（fullモデルを使用）
const createPoseLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
      delegate: "GPU"
    },
    runningMode: runningMode,
    numPoses: 2
  });
  demosSection.classList.remove("invisible");
};

createPoseLandmarker();

// Webカメラとキャンバスの取得
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

// Webカメラの利用が可能かチェックし、ボタンイベントを登録
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  const enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

function enableCam(event) {
  if (!poseLandmarker) {
    console.log("Wait! poseLandmarker not loaded yet.");
    return;
  }
  webcamRunning = !webcamRunning;
  const enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.innerText = webcamRunning ? "Stop Recording" : "Recording Start";

  const constraints = { video: facingMode: ideal:"enviroment"}}
};
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

// === 角度計算用ユーティリティ関数 ===

/**
 * 三点 A, B, C のうち B を頂点とした角度を算出（度数に変換）
 * @param {object} A {x, y}
 * @param {object} B {x, y}
 * @param {object} C {x, y}
 * @returns {number} 角度（度）
 */
function calculateAngle(A, B, C) {
  const AB = { x: A.x - B.x, y: A.y - B.y };
  const CB = { x: C.x - B.x, y: C.y - B.y };
  const dot = AB.x * CB.x + AB.y * CB.y;
  const normAB = Math.sqrt(AB.x * AB.x + AB.y * AB.y);
  const normCB = Math.sqrt(CB.x * CB.x + CB.y * CB.y);
  const angleRad = Math.acos(dot / (normAB * normCB));
  return angleRad * (180 / Math.PI);
}

/**
 * ベクトルと垂直（(0, -1)）とのなす角を算出（度数に変換）
 * ※画像上では y 座標が下方向に増加するため、垂直は (0, -1) と定義
 * @param {object} vector {x, y}
 * @returns {number} 角度（度）
 */
function angleWithVertical(vector) {
  const vertical = { x: 0, y: -1 };
  const dot = vector.x * vertical.x + vector.y * vertical.y;
  const normVector = Math.sqrt(vector.x * vector.x + vector.y * vector.y);
  const angleRad = Math.acos(dot / normVector);
  return angleRad * (180 / Math.PI);
}

/**
 * MediaPipe Pose のランドマークから、全ての関節角度を算出する関数  
 * 撮影側は体幹前後屈の算出にのみ用い、肩と肘は左右両側で算出します。
 *
 * 使用するランドマークインデックス：
 *  - 鼻: 0  
 *  - 左肩: 11, 左肘: 13, 左手首: 15, 左人差し指: 19, 左股関節: 23, 左膝: 25, 左足首: 27  
 *  - 右肩: 12, 右肘: 14, 右手首: 16, 右人差し指: 20, 右股関節: 24, 右膝: 26, 右足首: 28
 *
 * さらに、体幹回旋は左右の肩と左右の股関節の角度差から算出します。
 *
 * @param {Array} landmarks MediaPipe Pose のランドマーク配列
 * @param {string} filmingSide "left" または "right"（体幹前後屈用に撮影側を指定、初期設定は "left"）
 * @returns {object|null} 各角度を含むオブジェクト（必要なランドマークがなければ null）
 */
function computeAllAngles(landmarks, filmingSide = "left") {
  // 必要なランドマークの存在チェック
  const requiredIndexes = [0, 11, 13, 15, 19, 23, 25, 27, 12, 14, 16, 20, 24, 26, 28];
  for (const idx of requiredIndexes) {
    if (!landmarks[idx]) {
      console.warn(`landmark ${idx} is missing.`);
      return null;
    }
  }
  
  // Neck Angle: 両肩の中点から鼻へのベクトルと垂直とのなす角
  const shoulderCenter = {
    x: (landmarks[11].x + landmarks[12].x) / 2,
    y: (landmarks[11].y + landmarks[12].y) / 2
  };
  const nose = landmarks[0];
  const neckVector = { x: nose.x - shoulderCenter.x, y: nose.y - shoulderCenter.y };
  const neckAngle = angleWithVertical(neckVector);

  // Trunk Flexion Angle: 撮影側の肩から同側股関節へのベクトルと垂直とのなす角
  let trunkShoulderIndex, trunkHipIndex;
  if (filmingSide === "left") {
    trunkShoulderIndex = 11;
    trunkHipIndex = 23;
  } else {
    trunkShoulderIndex = 12;
    trunkHipIndex = 24;
  }
  const trunkVector = {
    x: landmarks[trunkHipIndex].x - landmarks[trunkShoulderIndex].x,
    y: landmarks[trunkHipIndex].y - landmarks[trunkShoulderIndex].y
  };
  const trunkFlexionAngle = angleWithVertical(trunkVector);

  // Trunk Rotation Angle: 左右の肩と左右の股関節の線の角度の差
  const shoulderRotationAngle = Math.atan2(landmarks[12].y - landmarks[11].y, landmarks[12].x - landmarks[11].x) * (180 / Math.PI);
  const hipRotationAngle = Math.atan2(landmarks[24].y - landmarks[23].y, landmarks[24].x - landmarks[23].x) * (180 / Math.PI);
  const trunkRotationAngle = shoulderRotationAngle - hipRotationAngle;

  // 左肩 Angle: 左肩を頂点に、左股関節と左肘のベクトルのなす角
  const leftShoulderAngle = calculateAngle(landmarks[23], landmarks[11], landmarks[13]);
  // 右肩 Angle: 右肩を頂点に、右股関節と右肘のベクトルのなす角
  const rightShoulderAngle = calculateAngle(landmarks[24], landmarks[12], landmarks[14]);

  // 左肘 Angle: 左肘を頂点に、左肩と左手首のベクトルのなす角
  const leftElbowAngle = calculateAngle(landmarks[11], landmarks[13], landmarks[15]);
  // 右肘 Angle: 右肘を頂点に、右肩と右手首のベクトルのなす角
  const rightElbowAngle = calculateAngle(landmarks[12], landmarks[14], landmarks[16]);

  // 左 Wrist Angle: 左肘を頂点に、左手首と左人差し指のなす角
  const leftWristAngle = calculateAngle(landmarks[13], landmarks[15], landmarks[19]);
  // 右 Wrist Angle: 右肘を頂点に、右手首と右人差し指のなす角
  const rightWristAngle = calculateAngle(landmarks[14], landmarks[16], landmarks[20]);

  // 左 Knee Angle: 左膝を頂点に、左股関節、左膝、左足首のなす角
  const leftKneeAngle = calculateAngle(landmarks[23], landmarks[25], landmarks[27]);
  // 右 Knee Angle: 右膝を頂点に、右股関節、右膝、右足首のなす角
  const rightKneeAngle = calculateAngle(landmarks[24], landmarks[26], landmarks[28]);

  return {
    neckAngle,
    trunkFlexionAngle,
    trunkRotationAngle,
    leftShoulderAngle,
    rightShoulderAngle,
    leftElbowAngle,
    rightElbowAngle,
    leftWristAngle,
    rightWristAngle,
    leftKneeAngle,
    rightKneeAngle
  };
}

/**
 * キャンバス上に各角度をテキスト表示する関数
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} angles 各角度を持つオブジェクト
 */
function displayAngles(ctx, angles) {
  ctx.fillStyle = "red";
  ctx.font = "16px Arial";
  const lineHeight = 18;
  let startX = 10;
  let startY = 20;
  const lines = [
    `Neck Angle: ${angles.neckAngle.toFixed(1)}°`,
    `Trunk Flexion: ${angles.trunkFlexionAngle.toFixed(1)}°`,
    `Trunk Rotation: ${angles.trunkRotationAngle.toFixed(1)}°`,
    `Left Shoulder: ${angles.leftShoulderAngle.toFixed(1)}°`,
    `Right Shoulder: ${angles.rightShoulderAngle.toFixed(1)}°`,
    `Left Elbow: ${angles.leftElbowAngle.toFixed(1)}°`,
    `Right Elbow: ${angles.rightElbowAngle.toFixed(1)}°`,
    `Left Wrist: ${angles.leftWristAngle.toFixed(1)}°`,
    `Right Wrist: ${angles.rightWristAngle.toFixed(1)}°`,
    `Left Knee: ${angles.leftKneeAngle.toFixed(1)}°`,
    `Right Knee: ${angles.rightKneeAngle.toFixed(1)}°`
  ];
  lines.forEach(line => {
    ctx.fillText(line, startX, startY);
    startY += lineHeight;
  });
}

// === Webカメラ映像からの検出と描画ループ ===
async function predictWebcam() {
  canvasElement.style.height = videoHeight;
  video.style.height = videoHeight;
  canvasElement.style.width = videoWidth;
  video.style.width = videoWidth;

  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await poseLandmarker.setOptions({ runningMode: "VIDEO" });
  }
  let startTimeMs = performance.now();
  poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    for (const landmarkSet of result.landmarks) {
      // ランドマークと接続線の描画
      drawingUtils.drawLandmarks(landmarkSet, {
        radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
      });
      drawingUtils.drawConnectors(landmarkSet, PoseLandmarker.POSE_CONNECTIONS);

      // 全関節角度の算出（撮影側は "left" を使用、必要に応じて "right" に変更可）
      const angles = computeAllAngles(landmarkSet, "left");
      if (angles) {
        console.log("Angles:", angles);
        displayAngles(canvasCtx, angles);
      } else {
        console.warn("Not all required landmarks detected.");
      }
    }
    canvasCtx.restore();
  });

  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}
