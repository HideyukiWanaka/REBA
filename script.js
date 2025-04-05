import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos") || document.body;
let poseLandmarker;
let runningMode = "IMAGE";
let webcamRunning = false;
const videoHeight = "360px";
const videoWidth = "480px";

// MediaPipe Pose Landmarker の初期化（fullモデル）
const createPoseLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
      delegate: "GPU"
    },
    runningMode: runningMode,
    numPoses: 2
  });
  demosSection.classList && demosSection.classList.remove("invisible");
};

createPoseLandmarker();

// 要素の取得
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

// Webカメラ起動
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

  // スマホ用：外向けカメラを優先
  const constraints = { video: { facingMode: { ideal: "environment" } } };
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

// === 角度計算用ユーティリティ ===

/**
 * 三点間の角度（Bを頂点）を算出（度数）
 */
function calculateAngle(A, B, C) {
  const AB = { x: A.x - B.x, y: A.y - B.y };
  const CB = { x: C.x - B.x, y: C.y - B.y };
  const dot = AB.x * CB.x + AB.y * CB.y;
  const normAB = Math.sqrt(AB.x ** 2 + AB.y ** 2);
  const normCB = Math.sqrt(CB.x ** 2 + CB.y ** 2);
  const angleRad = Math.acos(dot / (normAB * normCB));
  return angleRad * (180 / Math.PI);
}

/**
 * ベクトルと垂直（(0,-1)）との角度を算出（度数）
 */
function angleWithVertical(vector) {
  const vertical = { x: 0, y: -1 };
  const dot = vector.x * vertical.x + vector.y * vertical.y;
  const normVector = Math.sqrt(vector.x ** 2 + vector.y ** 2);
  const angleRad = Math.acos(dot / normVector);
  return angleRad * (180 / Math.PI);
}

/**
 * MediaPipe Poseのランドマークから、各関節角度を算出する
 * 撮影側は"left"（体幹前後屈に使用）
 */
function computeAllAngles(landmarks, filmingSide = "left") {
  const required = [0, 11, 13, 15, 19, 23, 25, 27, 12, 14, 16, 20, 24, 26, 28];
  for (const idx of required) {
    if (!landmarks[idx]) {
      console.warn(`landmark ${idx} is missing.`);
      return null;
    }
  }
  
  // 首角度：両肩中点→鼻
  const shoulderCenter = {
    x: (landmarks[11].x + landmarks[12].x) / 2,
    y: (landmarks[11].y + landmarks[12].y) / 2
  };
  const nose = landmarks[0];
  const neckVector = { x: nose.x - shoulderCenter.x, y: nose.y - shoulderCenter.y };
  const neckAngle = angleWithVertical(neckVector);
  
  // 体幹前後屈：撮影側の肩→股関節
  let trunkShoulderIndex, trunkHipIndex;
  if (filmingSide === "left") { trunkShoulderIndex = 11; trunkHipIndex = 23; }
  else { trunkShoulderIndex = 12; trunkHipIndex = 24; }
  const trunkVector = {
    x: landmarks[trunkHipIndex].x - landmarks[trunkShoulderIndex].x,
    y: landmarks[trunkHipIndex].y - landmarks[trunkShoulderIndex].y
  };
  const trunkFlexionAngle = angleWithVertical(trunkVector);
  
  // 体幹回旋：左右の肩線と股関節線の角度差
  const shoulderRotationAngle = Math.atan2(landmarks[12].y - landmarks[11].y, landmarks[12].x - landmarks[11].x) * (180 / Math.PI);
  const hipRotationAngle = Math.atan2(landmarks[24].y - landmarks[23].y, landmarks[24].x - landmarks[23].x) * (180 / Math.PI);
  const trunkRotationAngle = shoulderRotationAngle - hipRotationAngle;
  
  // 左肩角度：左肩を頂点、左股関節→左肘
  const leftShoulderAngle = calculateAngle(landmarks[23], landmarks[11], landmarks[13]);
  // 右肩角度：右肩を頂点、右股関節→右肘
  const rightShoulderAngle = calculateAngle(landmarks[24], landmarks[12], landmarks[14]);
  
  // 左肘角度：左肘を頂点、左肩→左手首
  const leftElbowAngle = calculateAngle(landmarks[11], landmarks[13], landmarks[15]);
  // 右肘角度：右肘を頂点、右肩→右手首
  const rightElbowAngle = calculateAngle(landmarks[12], landmarks[14], landmarks[16]);
  
  // 左手首角度：左肘を頂点、左手首→左人差し指
  const leftWristAngle = calculateAngle(landmarks[13], landmarks[15], landmarks[19]);
  // 右手首角度：右肘を頂点、右手首→右人差し指
  const rightWristAngle = calculateAngle(landmarks[14], landmarks[16], landmarks[20]);
  
  // 左膝角度：左膝を頂点、左股関節→左足首
  const leftKneeAngle = calculateAngle(landmarks[23], landmarks[25], landmarks[27]);
  // 右膝角度：右膝を頂点、右股関節→右足首
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

// 外部補正値をラジオボタンから取得
function getCalibrationInputs() {
  return {
    neckRotation: Number(document.querySelector('input[name="neckRotation"]:checked').value),
    neckLateralBending: Number(document.querySelector('input[name="neckLateralBending"]:checked').value),
    trunkLateralFlexion: Number(document.querySelector('input[name="trunkLateralFlexion"]:checked').value),
    loadForce: Number(document.querySelector('input[name="loadForce"]:checked').value),
    coupling: Number(document.querySelector('input[name="coupling"]:checked').value),
    activityScore: Number(document.querySelector('input[name="activityScore"]:checked').value),
    postureType: document.querySelector('input[name="postureType"]:checked').value
  };
}

// 補正済み角度の統合：膝角度はSitting/Walkingの場合は1固定
function calibrateREBAAngles(computedAngles, calibInputs) {
  let finalLeftKneeAngle = computedAngles.leftKneeAngle;
  let finalRightKneeAngle = computedAngles.rightKneeAngle;
  if (calibInputs.postureType === "sitting" || calibInputs.postureType === "walking") {
    finalLeftKneeAngle = 1;
    finalRightKneeAngle = 1;
  }
  return {
    // 補正項目（外部入力）
    neckRotation: calibInputs.neckRotation,
    neckLateralBending: calibInputs.neckLateralBending,
    trunkLateralFlexion: calibInputs.trunkLateralFlexion,
    loadForce: calibInputs.loadForce,
    coupling: calibInputs.coupling,
    activityScore: calibInputs.activityScore,
    // 自動測定された角度
    neckAngle: computedAngles.neckAngle,
    trunkFlexionAngle: computedAngles.trunkFlexionAngle,
    trunkRotationAngle: computedAngles.trunkRotationAngle,
    leftShoulderAngle: computedAngles.leftShoulderAngle,
    rightShoulderAngle: computedAngles.rightShoulderAngle,
    leftElbowAngle: computedAngles.leftElbowAngle,
    rightElbowAngle: computedAngles.rightElbowAngle,
    leftWristAngle: computedAngles.leftWristAngle,
    rightWristAngle: computedAngles.rightWristAngle,
    leftKneeAngle: finalLeftKneeAngle,
    rightKneeAngle: finalRightKneeAngle
  };
}

// グループAスコア算出（首・体幹・脚）
// 文献に基づく閾値を使用
function getGroupAScore(angles, calibInputs) {
  // 頸部スコア
  let neckScore = (angles.neckAngle < 10 ? 1 : angles.neckAngle < 20 ? 2 : 3);
  // 体幹前後屈スコア
  let trunkScore = (angles.trunkFlexionAngle < 20 ? 1 : angles.trunkFlexionAngle < 60 ? 2 : 3);
  // 体幹回旋スコア（絶対値で判定）
  let rotationScore = (Math.abs(angles.trunkRotationAngle) < 5 ? 1 : Math.abs(angles.trunkRotationAngle) < 15 ? 2 : 3);
  // 膝スコア（Standingの場合のみ、Sitting/Walkingは1固定）
  let kneeScore = 0;
  if (calibInputs.postureType === "sitting" || calibInputs.postureType === "walking") {
    kneeScore = 1;
  } else {
    let avgKnee = (angles.leftKneeAngle + angles.rightKneeAngle) / 2;
    kneeScore = (avgKnee < 90 ? 3 : avgKnee < 120 ? 2 : 1);
  }
  // 荷重補正：5kg→0, 10kg→+1, 15kg→+2
  let loadCorrection = (calibInputs.loadForce === 5 ? 0 : calibInputs.loadForce === 10 ? 1 : 2);
  
  return neckScore + trunkScore + rotationScore + kneeScore + loadCorrection;
}

// グループBスコア算出（上腕・下腕・手首）
// 文献の閾値を使用
function getGroupBScore(angles, calibInputs) {
  // 上腕スコア（左右）
  let leftUpperArmScore = (angles.leftShoulderAngle < 20 ? 1 : angles.leftShoulderAngle < 45 ? 2 : 3);
  let rightUpperArmScore = (angles.rightShoulderAngle < 20 ? 1 : angles.rightShoulderAngle < 45 ? 2 : 3);
  // 下腕（肘）スコア（左右）：肘角度が小さい＝伸展、点数が低い
  let leftElbowScore = (angles.leftElbowAngle < 60 ? 1 : angles.leftElbowAngle < 100 ? 2 : 3);
  let rightElbowScore = (angles.rightElbowAngle < 60 ? 1 : angles.rightElbowAngle < 100 ? 2 : 3);
  // 手首スコア（左右）
  let leftWristScore = (angles.leftWristAngle < 15 ? 1 : angles.leftWristAngle < 30 ? 2 : 3);
  let rightWristScore = (angles.rightWristAngle < 15 ? 1 : angles.rightWristAngle < 30 ? 2 : 3);
  
  // 把持安定性の補正値（文献通り：1～3）
  let couplingCorrection = calibInputs.coupling;
  
  return leftUpperArmScore + rightUpperArmScore + leftElbowScore + rightElbowScore + leftWristScore + rightWristScore + couplingCorrection;
}

// 総合REBAスコア算出（グループA + グループB + 活動性）
function getFinalREBAScore(angles, calibInputs) {
  let groupAScore = getGroupAScore(angles, calibInputs);
  let groupBScore = getGroupBScore(angles, calibInputs);
  let activityScore = calibInputs.activityScore; // 1～3
  return groupAScore + groupBScore + activityScore;
}

// リスクレベル判定（文献に準じた基準）
function getRiskLevel(score) {
  if (score <= 3) return "Negligible";
  else if (score <= 7) return "Low";
  else if (score <= 10) return "Medium";
  else if (score <= 13) return "High";
  else return "Very High";
}

// キャンバス上に角度・スコア・リスクを表示
function displayAngles(ctx, angles, finalScore, riskLevel) {
  ctx.fillStyle = "red";
  ctx.font = "16px Arial";
  const lineHeight = 18;
  let startX = 10, startY = 20;
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
    `Right Knee: ${angles.rightKneeAngle.toFixed(1)}°`,
    "-----------------------",
    `REBA Score: ${finalScore}`,
    `Risk Level: ${riskLevel}`
  ];
  lines.forEach(line => {
    ctx.fillText(line, startX, startY);
    startY += lineHeight;
  });
}

// 検出ループ
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
      drawingUtils.drawLandmarks(landmarkSet, {
        radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
      });
      drawingUtils.drawConnectors(landmarkSet, PoseLandmarker.POSE_CONNECTIONS);
      
      const computedAngles = computeAllAngles(landmarkSet, "left"); // 撮影側は "left" で採用
      if (computedAngles) {
        const calibInputs = getCalibrationInputs();
        const calibratedAngles = calibrateREBAAngles(computedAngles, calibInputs);
        const finalScore = getFinalREBAScore(calibratedAngles, calibInputs);
        const riskLevel = getRiskLevel(finalScore);
        console.log("Calibrated Angles:", calibratedAngles, "Final Score:", finalScore, "Risk:", riskLevel);
        displayAngles(canvasCtx, calibratedAngles, finalScore, riskLevel);
      } else {
        console.warn("必要なランドマークがすべて検出できませんでした。");
      }
    }
    canvasCtx.restore();
  });
  
  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}