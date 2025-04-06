import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos") || document.body;
let poseLandmarker;
let runningMode = "IMAGE";
let webcamRunning = false;
const videoHeight = "360px";
const videoWidth = "480px";

// MediaPipe Pose Landmarker（fullモデル）の初期化
async function initPoseLandmarker() {
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
      delegate: "GPU"
    },
    runningMode,
    numPoses: 2,
  });
  demosSection.classList && demosSection.classList.remove("invisible");
}
initPoseLandmarker();

// 要素取得
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

// Webカメラ起動
document.getElementById("webcamButton").addEventListener("click", () => {
  if (!poseLandmarker) return;
  webcamRunning = !webcamRunning;
  document.getElementById("webcamButton").innerText = webcamRunning ? "Stop Recording" : "Recording Start";
  navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
      video.srcObject = stream;
      video.addEventListener("loadeddata", predictWebcam);
    })
    .catch((err) => console.error(err));
});

// ===== 角度計算ユーティリティ =====
function calculateAngle(A, B, C) {
  const AB = { x: A.x - B.x, y: A.y - B.y };
  const CB = { x: C.x - B.x, y: C.y - B.y };
  const dot = AB.x * CB.x + AB.y * CB.y;
  const normAB = Math.sqrt(AB.x ** 2 + AB.y ** 2);
  const normCB = Math.sqrt(CB.x ** 2 + CB.y ** 2);
  const angleRad = Math.acos(dot / (normAB * normCB));
  return (angleRad * 180) / Math.PI;
}

function angleWithVertical(vector) {
  const vertical = { x: 0, y: -1 };
  const dot = vector.x * vertical.x + vector.y * vertical.y;
  const norm = Math.sqrt(vector.x ** 2 + vector.y ** 2);
  const angleRad = Math.acos(dot / norm);
  return (angleRad * 180) / Math.PI;
}

// ===== MediaPipe Pose から各関節角度算出 =====
function computeAllAngles(landmarks, filmingSide = "left") {
  const required = [0, 11, 13, 15, 19, 23, 25, 27, 12, 14, 16, 20, 24, 26, 28];
  for (const idx of required) {
    if (!landmarks[idx]) {
      console.warn(`landmark ${idx} is missing.`);
      return null;
    }
  }
  const shoulderCenter = {
    x: (landmarks[11].x + landmarks[12].x) / 2,
    y: (landmarks[11].y + landmarks[12].y) / 2
  };
  const nose = landmarks[0];
  const neckVector = { x: nose.x - shoulderCenter.x, y: nose.y - shoulderCenter.y };
  const neckAngle = angleWithVertical(neckVector);
  
  let trunkShoulderIndex, trunkHipIndex;
  if (filmingSide === "left") { trunkShoulderIndex = 11; trunkHipIndex = 23; }
  else { trunkShoulderIndex = 12; trunkHipIndex = 24; }
  const trunkVector = {
    x: landmarks[trunkHipIndex].x - landmarks[trunkShoulderIndex].x,
    y: landmarks[trunkHipIndex].y - landmarks[trunkShoulderIndex].y
  };
  const trunkFlexionAngle = angleWithVertical(trunkVector);
  
  const shoulderRotationAngle = Math.atan2(landmarks[12].y - landmarks[11].y, landmarks[12].x - landmarks[11].x) * (180 / Math.PI);
  const hipRotationAngle = Math.atan2(landmarks[24].y - landmarks[23].y, landmarks[24].x - landmarks[23].x) * (180 / Math.PI);
  const trunkRotationAngle = shoulderRotationAngle - hipRotationAngle;
  
  const leftShoulderAngle = calculateAngle(landmarks[23], landmarks[11], landmarks[13]);
  const rightShoulderAngle = calculateAngle(landmarks[24], landmarks[12], landmarks[14]);
  
  const leftElbowAngle = calculateAngle(landmarks[11], landmarks[13], landmarks[15]);
  const rightElbowAngle = calculateAngle(landmarks[12], landmarks[14], landmarks[16]);
  
  const leftWristAngle = calculateAngle(landmarks[13], landmarks[15], landmarks[19]);
  const rightWristAngle = calculateAngle(landmarks[14], landmarks[16], landmarks[20]);
  
  const leftKneeAngle = calculateAngle(landmarks[23], landmarks[25], landmarks[27]);
  const rightKneeAngle = calculateAngle(landmarks[24], landmarks[26], landmarks[28]);
  
  // 上腕角度：肩関節の屈曲角度（例：左側）
  const upperArmAngle = calculateAngle(landmarks[23], landmarks[11], landmarks[13]);
  // 前腕角度：肘関節の屈曲角度
  const forearmAngle = calculateAngle(landmarks[11], landmarks[13], landmarks[15]);
  
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
    rightKneeAngle,
    upperArmAngle,
    forearmAngle
  };
}

// ===== 外部補正入力の取得 =====
function getCalibrationInputs() {
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

// ===== 補正済み角度の統合 =====
function calibrateREBAAngles(computedAngles, calibInputs) {
  let finalLeftKneeAngle = computedAngles.leftKneeAngle;
  let finalRightKneeAngle = computedAngles.rightKneeAngle;
  if (calibInputs.postureCategory === "sittingWalking") {
    finalLeftKneeAngle = 1;
    finalRightKneeAngle = 1;
  }
  return {
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
    rightKneeAngle: finalRightKneeAngle,
    neckRotation: calibInputs.neckRotation,
    neckLateralBending: calibInputs.neckLateralBending,
    trunkLateralFlexion: calibInputs.trunkLateralFlexion,
    loadForce: calibInputs.loadForce,
    postureCategory: calibInputs.postureCategory,
    upperArmCorrection: calibInputs.upperArmCorrection,
    shoulderElevation: calibInputs.shoulderElevation,
    gravityAssist: calibInputs.gravityAssist,
    wristCorrection: calibInputs.wristCorrection,
    staticPosture: calibInputs.staticPosture,
    repetitiveMovement: calibInputs.repetitiveMovement,
    unstableMovement: calibInputs.unstableMovement,
    coupling: calibInputs.coupling
  };
}

// ===== Table A, B, C のルックアップ =====

// Table A: キー "T,N,L" (体幹, 頸, 下肢)
const tableA_Lookup = {
  "1,1,1": 1, "1,1,2": 2, "1,1,3": 3, "1,1,4": 4,
  "1,2,1": 1, "1,2,2": 2, "1,2,3": 3, "1,2,4": 4,
  "1,3,1": 3, "1,3,2": 3, "1,3,3": 5, "1,3,4": 6,
  "2,1,1": 2, "2,1,2": 3, "2,1,3": 4, "2,1,4": 5,
  "2,2,1": 3, "2,2,2": 4, "2,2,3": 5, "2,2,4": 6,
  "2,3,1": 4, "2,3,2": 5, "2,3,3": 6, "2,3,4": 7,
  "3,1,1": 2, "3,1,2": 4, "3,1,3": 5, "3,1,4": 6,
  "3,2,1": 4, "3,2,2": 5, "3,2,3": 6, "3,2,4": 7,
  "3,3,1": 5, "3,3,2": 6, "3,3,3": 7, "3,3,4": 8,
  "4,1,1": 3, "4,1,2": 5, "4,1,3": 6, "4,1,4": 7,
  "4,2,1": 5, "4,2,2": 6, "4,2,3": 7, "4,2,4": 8,
  "4,3,1": 6, "4,3,2": 7, "4,3,3": 8, "4,3,4": 9,
  "5,1,1": 4, "5,1,2": 6, "5,1,3": 7, "5,1,4": 8,
  "5,2,1": 6, "5,2,2": 7, "5,2,3": 8, "5,2,4": 9,
  "5,3,1": 7, "5,3,2": 8, "5,3,3": 9, "5,3,4": 9
};

function getScoreA(trunk, neck, leg, loadKg) {
  const key = `${trunk},${neck},${leg}`;
  const base = tableA_Lookup[key] ?? 1;
  return base + calcLoadScore(loadKg);
}

// Table B: キー "U,F,W" (上腕, 前腕, 手首)
// 上腕: 1～6, 前腕: 1～2, 手首: 1～3 として ScoreB = (上腕 + 前腕 + 手首) - 2 を基本計算する
const tableB_Lookup = {
  "1,1,1": 1, "1,1,2": 2, "1,1,3": 2,
  "1,2,1": 1, "1,2,2": 2, "1,2,3": 3,
  "2,1,1": 1, "2,1,2": 2, "2,1,3": 3,
  "2,2,1": 2, "2,2,2": 3, "2,2,3": 4,
  "3,1,1": 3, "3,1,2": 4, "3,1,3": 5,
  "3,2,1": 4, "3,2,2": 5, "3,2,3": 5,
  "4,1,1": 4, "4,1,2": 5, "4,1,3": 5,
  "4,2,1": 5, "4,2,2": 6, "4,2,3": 7,
  "5,1,1": 6, "5,1,2": 7, "5,1,3": 8,
  "5,2,1": 7, "5,2,2": 8, "5,2,3": 8,
  "6,1,1": 7, "6,1,2": 8, "6,1,3": 8,
  "6,2,1": 8, "6,2,2": 9, "6,2,3": 9
};

function getScoreB(upperArm, forearm, wrist, coupling) {
  const key = `${upperArm},${forearm},${wrist}`;
  const base = tableB_Lookup[key] ?? 1;
  return base + coupling;
}

// Table C: 組み合わせ表 (ScoreA, ScoreB) → 統合スコア (1～12)
// 表3に基づく定義
const tableC_Lookup = {
  "1,1": 1, "1,2": 1, "1,3": 1, "1,4": 2, "1,5": 3, "1,6": 3, "1,7": 4, "1,8": 5, "1,9": 6, "1,10": 7, "1,11": 7, "1,12": 7,
  "2,1": 1, "2,2": 2, "2,3": 2, "2,4": 3, "2,5": 4, "2,6": 4, "2,7": 5, "2,8": 6, "2,9": 6, "2,10": 7, "2,11": 7, "2,12": 8,
  "3,1": 2, "3,2": 3, "3,3": 3, "3,4": 4, "3,5": 5, "3,6": 6, "3,7": 7, "3,8": 7, "3,9": 7, "3,10": 8, "3,11": 8, "3,12": 8,
  "4,1": 3, "4,2": 4, "4,3": 4, "4,4": 4, "4,5": 5, "4,6": 6, "4,7": 7, "4,8": 8, "4,9": 8, "4,10": 9, "4,11": 9, "4,12": 9,
  "5,1": 4, "5,2": 4, "5,3": 4, "5,4": 5, "5,5": 6, "5,6": 7, "5,7": 8, "5,8": 8, "5,9": 9, "5,10": 9, "5,11": 9, "5,12": 9,
  "6,1": 6, "6,2": 6, "6,3": 6, "6,4": 7, "6,5": 8, "6,6": 8, "6,7": 9, "6,8": 9, "6,9": 10, "6,10": 10, "6,11": 10, "6,12": 10,
  "7,1": 7, "7,2": 7, "7,3": 7, "7,4": 8, "7,5": 9, "7,6": 9, "7,7": 9, "7,8": 10, "7,9": 10, "7,10": 11, "7,11": 11, "7,12": 11,
  "8,1": 8, "8,2": 8, "8,3": 8, "8,4": 9, "8,5": 10, "8,6": 10, "8,7": 10, "8,8": 10, "8,9": 11, "8,10": 11, "8,11": 11, "8,12": 12,
  "9,1": 9, "9,2": 9, "9,3": 9, "9,4": 10, "9,5": 10, "9,6": 11, "9,7": 11, "9,8": 11, "9,9": 12, "9,10": 12, "9,11": 12, "9,12": 12,
  "10,1": 10, "10,2": 10, "10,3": 10, "10,4": 11, "10,5": 11, "10,6": 11, "10,7": 12, "10,8": 12, "10,9": 12, "10,10": 12, "10,11": 12, "10,12": 12,
  "11,1": 11, "11,2": 11, "11,3": 11, "11,4": 12, "11,5": 12, "11,6": 12, "11,7": 12, "11,8": 12, "11,9": 12, "11,10": 12, "11,11": 12, "11,12": 12,
  "12,1": 12, "12,2": 12, "12,3": 12, "12,4": 12, "12,5": 12, "12,6": 12, "12,7": 12, "12,8": 12, "12,9": 12, "12,10": 12, "12,11": 12, "12,12": 12
};

function getTableCScore(scoreA, scoreB) {
  const key = `${scoreA},${scoreB}`;
  return tableC_Lookup[key] || 1;
}

// ===== 各部位の評価関数 =====

function evaluateTrunk(computedAngles, calibInputs) {
  const rotationFlag = Math.abs(computedAngles.trunkRotationAngle) >= 10;
  const sideBendFlag = (calibInputs.trunkLateralFlexion > 0);
  return calcTrunkScore(computedAngles.trunkFlexionAngle, rotationFlag, sideBendFlag);
}

function evaluateNeck(computedAngles, calibInputs) {
  const rotationFlag = (calibInputs.neckRotation > 0);
  const sideBendFlag = (calibInputs.neckLateralBending > 0);
  return calcNeckScore(computedAngles.neckAngle, rotationFlag, sideBendFlag);
}

function calcLegScoreUnified(postureCategory, kneeFlexAngle) {
  if (postureCategory === "sittingWalking") return 1;
  const base = (postureCategory === "standingBoth") ? 1 : 2;
  let add = 0;
  if (kneeFlexAngle >= 30 && kneeFlexAngle <= 60) add = 1;
  else if (kneeFlexAngle > 60) add = 2;
  return base + add;
}

function evaluateLeg(computedAngles, calibInputs) {
  const avgKnee = (computedAngles.leftKneeAngle + computedAngles.rightKneeAngle) / 2;
  return calcLegScoreUnified(calibInputs.postureCategory, avgKnee);
}

function evaluateUpperArm(computedAngles, calibInputs) {
  const avg = (computedAngles.leftShoulderAngle + computedAngles.rightShoulderAngle) / 2;
  return calcUpperArmScore(avg, calibInputs.upperArmCorrection, calibInputs.shoulderElevation, calibInputs.gravityAssist);
}

function evaluateForearm(computedAngles) {
  const avg = (computedAngles.leftElbowAngle + computedAngles.rightElbowAngle) / 2;
  return calcForearmScore(avg);
}

function evaluateWrist(computedAngles, calibInputs) {
  const avg = (computedAngles.leftWristAngle + computedAngles.rightWristAngle) / 2;
  return calcWristScore(avg, calibInputs.wristCorrection);
}

function evaluateActivity(calibInputs) {
  return calibInputs.staticPosture + calibInputs.repetitiveMovement + calibInputs.unstableMovement;
}

// ===== 補正用計算関数 =====

function calcNeckScore(neckAngle, rotationFlag, sideBendFlag) {
  const base = (neckAngle <= 20 ? 1 : 2);
  return base + (rotationFlag || sideBendFlag ? 1 : 0);
}

function calcTrunkScore(trunkFlexAngle, rotationFlag, sideBendFlag) {
  let base;
  if (trunkFlexAngle === 0) base = 1;
  else if (trunkFlexAngle <= 20) base = 2;
  else if (trunkFlexAngle <= 60) base = 3;
  else base = 4;
  return base + (rotationFlag || sideBendFlag ? 1 : 0);
}

function calcLegScoreUnified(postureCategory, kneeFlexAngle) {
  if (postureCategory === "sittingWalking") return 1;
  const base = (postureCategory === "standingBoth") ? 1 : 2;
  let add = 0;
  if (kneeFlexAngle >= 30 && kneeFlexAngle <= 60) add = 1;
  else if (kneeFlexAngle > 60) add = 2;
  return base + add;
}

function calcUpperArmScore(upperArmAngle, upperArmCorrection, shoulderElevation, gravityAssist) {
  const absAngle = Math.abs(upperArmAngle);
  let base;
  if (absAngle <= 20) base = 1;
  else if (absAngle <= 45) base = 2;
  else if (absAngle <= 90) base = 3;
  else base = 4;
  // 補正スコアを加え、最終的に最大6点になるようにする
  const corrected = base + upperArmCorrection + shoulderElevation + gravityAssist;
  if (corrected < 1) return 1;
  if (corrected > 6) return 6;
  return corrected;
}

function calcForearmScore(elbowAngle) {
  return (elbowAngle >= 60 && elbowAngle <= 100) ? 1 : 2;
}

function calcWristScore(wristAngle, wristCorrection) {
  const base = (wristAngle > 15) ? 2 : 1;
  const score = base + wristCorrection;
  if (score < 1) return 1;
  if (score > 3) return 3;
  return score;
}

function calcLoadScore(loadKg) {
  if (loadKg < 5) return 0;
  else if (loadKg <= 10) return 1;
  else return 2;
}

// ===== 最終REBAスコア算出 =====
function getFinalREBAScore(computedAngles, calibInputs) {
  const trunkScore = evaluateTrunk(computedAngles, calibInputs);
  const neckScore = evaluateNeck(computedAngles, calibInputs);
  const legScore = evaluateLeg(computedAngles, calibInputs);
  const scoreA = getScoreA(trunkScore, neckScore, legScore, calibInputs.loadForce);
  
  const upperArmScore = evaluateUpperArm(computedAngles, calibInputs);
  const forearmScore = evaluateForearm(computedAngles);
  const wristScore = evaluateWrist(computedAngles, calibInputs);
  const scoreB = getScoreB(upperArmScore, forearmScore, wristScore, calibInputs.coupling);
  
  const tableCScore = getTableCScore(scoreA, scoreB);
  const activityScore = evaluateActivity(calibInputs);
  
  let finalScore = tableCScore + activityScore;
  if (finalScore > 15) finalScore = 15;
  if (finalScore < 1) finalScore = 1;
  return finalScore;
}

function getRiskLevel(score) {
  if (score === 1) return "無視できる";
  else if (score >= 2 && score <= 3) return "低リスク";
  else if (score >= 4 && score <= 7) return "中リスク";
  else if (score >= 8 && score <= 10) return "高リスク";
  else return "非常に高リスク";
}

function displayResults(ctx, computedAngles, finalScore, riskLevel) {
  ctx.fillStyle = "red";
  ctx.font = "16px Arial";
  const lineHeight = 18;
  let startX = 10, startY = 20;
  const lines = [
    `Neck Angle: ${computedAngles.neckAngle.toFixed(1)}°`,
    `Trunk Flexion: ${computedAngles.trunkFlexionAngle.toFixed(1)}°`,
    `Trunk Rotation: ${computedAngles.trunkRotationAngle.toFixed(1)}°`,
    `Left Shoulder: ${computedAngles.leftShoulderAngle.toFixed(1)}°`,
    `Right Shoulder: ${computedAngles.rightShoulderAngle.toFixed(1)}°`,
    `Left Elbow: ${computedAngles.leftElbowAngle.toFixed(1)}°`,
    `Right Elbow: ${computedAngles.rightElbowAngle.toFixed(1)}°`,
    `Left Wrist: ${computedAngles.leftWristAngle.toFixed(1)}°`,
    `Right Wrist: ${computedAngles.rightWristAngle.toFixed(1)}°`,
    `Left Knee: ${computedAngles.leftKneeAngle.toFixed(1)}°`,
    `Right Knee: ${computedAngles.rightKneeAngle.toFixed(1)}°`,
    "---------------------",
    `Final REBA Score: ${finalScore}`,
    `Risk Level: ${riskLevel}`
  ];
  lines.forEach(line => {
    ctx.fillText(line, startX, startY);
    startY += lineHeight;
  });
}

// ===== Webカメラからの検出ループ =====
async function predictWebcam() {
  canvasElement.style.height = videoHeight;
  video.style.height = videoHeight;
  canvasElement.style.width = videoWidth;
  video.style.width = videoWidth;
  
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await poseLandmarker.setOptions({ runningMode: "VIDEO" });
  }
  
  const startTimeMs = performance.now();
  poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    for (const landmarkSet of result.landmarks) {
      drawingUtils.drawLandmarks(landmarkSet, {
        radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
      });
      drawingUtils.drawConnectors(landmarkSet, PoseLandmarker.POSE_CONNECTIONS);
      
      const computedAngles = computeAllAngles(landmarkSet, getCalibrationInputs().filmingSide);
      if (computedAngles) {
        const calibInputs = getCalibrationInputs();
        const finalScore = getFinalREBAScore(computedAngles, calibInputs);
        const riskLevel = getRiskLevel(finalScore);
        console.log("Computed Angles:", computedAngles, "Final Score:", finalScore, "Risk Level:", riskLevel);
        displayResults(canvasCtx, computedAngles, finalScore, riskLevel);
      } else {
        console.warn("必要なランドマークがすべて検出できませんでした。");
      }
    }
  });
  
  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}