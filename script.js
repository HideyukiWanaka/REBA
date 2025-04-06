.import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos") || document.body;
let poseLandmarker;
let runningMode = "IMAGE";
let webcamRunning = false;
const videoHeight = "360px";
const videoWidth = "480px";

// MediaPipe Pose Landmarker（fullモデル）の初期化
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

// 要素取得
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

function enableCam() {
  if (!poseLandmarker) {
    console.log("Wait! poseLandmarker not loaded yet.");
    return;
  }
  webcamRunning = !webcamRunning;
  const enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.innerText = webcamRunning ? "Stop Recording" : "Recording Start";
  
  const constraints = { video: { facingMode: { ideal: "environment" } } };
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

// ===== 角度計算ユーティリティ =====

function calculateAngle(A, B, C) {
  const AB = { x: A.x - B.x, y: A.y - B.y };
  const CB = { x: C.x - B.x, y: C.y - B.y };
  const dot = AB.x * CB.x + AB.y * CB.y;
  const normAB = Math.sqrt(AB.x ** 2 + AB.y ** 2);
  const normCB = Math.sqrt(CB.x ** 2 + CB.y ** 2);
  const angleRad = Math.acos(dot / (normAB * normCB));
  return angleRad * (180 / Math.PI);
}

function angleWithVertical(vector) {
  const vertical = { x: 0, y: -1 };
  const dot = vector.x * vertical.x + vector.y * vertical.y;
  const norm = Math.sqrt(vector.x ** 2 + vector.y ** 2);
  const angleRad = Math.acos(dot / norm);
  return angleRad * (180 / Math.PI);
}

// ===== MediaPipe Pose から各関節角度算出 =====
// filmingSide ("left" or "right") により体幹計算に反映
function computeAllAngles(landmarks, filmingSide = "left") {
  const required = [0,11,13,15,19,23,25,27,12,14,16,20,24,26,28];
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
  
  // 体幹回旋：肩線と股関節線の角度差
  const shoulderRotationAngle = Math.atan2(landmarks[12].y - landmarks[11].y, landmarks[12].x - landmarks[11].x) * (180/Math.PI);
  const hipRotationAngle = Math.atan2(landmarks[24].y - landmarks[23].y, landmarks[24].x - landmarks[23].x) * (180/Math.PI);
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

// ===== 外部補正入力の取得 =====
function getCalibrationInputs() {
  return {
    filmingSide: document.querySelector('input[name="filmingSide"]:checked').value,
    neckRotation: Number(document.querySelector('input[name="neckRotation"]:checked').value),
    neckLateralBending: Number(document.querySelector('input[name="neckLateralBending"]:checked').value),
    trunkLateralFlexion: Number(document.querySelector('input[name="trunkLateralFlexion"]:checked').value),
    loadForce: Number(document.querySelector('input[name="loadForce"]:checked').value),
    weightBearing: document.querySelector('input[name="weightBearing"]:checked').value,
    upperArmCorrection: Number(document.querySelector('input[name="upperArmCorrection"]:checked').value),
    shoulderElevation: Number(document.querySelector('input[name="shoulderElevation"]:checked').value),
    gravityAssist: Number(document.querySelector('input[name="gravityAssist"]:checked').value),
    wristCorrection: Number(document.querySelector('input[name="wristCorrection"]:checked').value),
    staticPosture: Number(document.querySelector('input[name="staticPosture"]:checked').value),
    repetitiveMovement: Number(document.querySelector('input[name="repetitiveMovement"]:checked').value),
    unstableMovement: Number(document.querySelector('input[name="unstableMovement"]:checked').value),
    postureType: document.querySelector('input[name="postureType"]:checked').value
  };
}

// ===== 補正済み角度の統合 =====
// 膝角度は Sitting/Walkingの場合は1固定
function calibrateREBAAngles(computedAngles, calibInputs) {
  let finalLeftKneeAngle = computedAngles.leftKneeAngle;
  let finalRightKneeAngle = computedAngles.rightKneeAngle;
  if (calibInputs.postureType === "sitting" || calibInputs.postureType === "walking") {
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
    // 外部補正項目
    neckRotation: calibInputs.neckRotation,
    neckLateralBending: calibInputs.neckLateralBending,
    trunkLateralFlexion: calibInputs.trunkLateralFlexion,
    loadForce: calibInputs.loadForce,
    weightBearing: calibInputs.weightBearing,
    upperArmCorrection: calibInputs.upperArmCorrection,
    shoulderElevation: calibInputs.shoulderElevation,
    gravityAssist: calibInputs.gravityAssist,
    wristCorrection: calibInputs.wristCorrection,
    staticPosture: calibInputs.staticPosture,
    repetitiveMovement: calibInputs.repetitiveMovement,
    unstableMovement: calibInputs.unstableMovement,
    postureType: calibInputs.postureType
  };
}

// ===== Table A, B, C のルックアップ =====

// Table A: キー "T,N,L" (体幹, 頸, 下肢) → 基礎点（ここでは例として全組み合わせを定義）
const tableA_Lookup = {
  "1,1,1":1, "1,1,2":2, "1,1,3":3, "1,1,4":4,
  "1,2,1":1, "1,2,2":2, "1,2,3":3, "1,2,4":4,
  "1,3,1":3, "1,3,2":3, "1,3,3":5, "1,3,4":6,
  "2,1,1":2, "2,1,2":3, "2,1,3":4, "2,1,4":5,
  "2,2,1":3, "2,2,2":4, "2,2,3":5, "2,2,4":6,
  "2,3,1":4, "2,3,2":5, "2,3,3":6, "2,3,4":7,
  "3,1,1":2, "3,1,2":4, "3,1,3":5, "3,1,4":6,
  "3,2,1":4, "3,2,2":5, "3,2,3":6, "3,2,4":7,
  "3,3,1":5, "3,3,2":6, "3,3,3":7, "3,3,4":8,
  "4,1,1":3, "4,1,2":5, "4,1,3":6, "4,1,4":7,
  "4,2,1":5, "4,2,2":6, "4,2,3":7, "4,2,4":8,
  "4,3,1":6, "4,3,2":7, "4,3,3":8, "4,3,4":9,
  "5,1,1":4, "5,1,2":6, "5,1,3":7, "5,1,4":8,
  "5,2,1":6, "5,2,2":7, "5,2,3":8, "5,2,4":9,
  "5,3,1":7, "5,3,2":8, "5,3,3":9, "5,3,4":9
};
// Table A最終スコア = min((T + N + L) - 2, 8) + 荷重スコア
function getScoreA(trunkScore, neckScore, legScore, loadForce) {
  let base = (trunkScore + neckScore + legScore) - 2;
  if (base > 8) base = 8;
  // calcLoadScore で荷重スコアを求める（衝撃はここでは false とする）
  let loadScore = (loadForce === 5 ? 0 : (loadForce === 10 ? 1 : 2));
  return base + loadScore;
}

// Table B: キー "U,F,W" (上腕, 前腕, 手首) → 基礎点
const tableB_Lookup = {
  "1,1,1":1, "1,1,2":2, "1,1,3":3,
  "1,2,1":2, "1,2,2":3, "1,2,3":4,
  "1,3,1":3, "1,3,2":4, "1,3,3":5,
  "2,1,1":2, "2,1,2":3, "2,1,3":4,
  "2,2,1":3, "2,2,2":4, "2,2,3":5,
  "2,3,1":4, "2,3,2":5, "2,3,3":6,
  "3,1,1":3, "3,1,2":4, "3,1,3":5,
  "3,2,1":4, "3,2,2":5, "3,2,3":6,
  "3,3,1":5, "3,3,2":6, "3,3,3":7,
  "4,1,1":4, "4,1,2":5, "4,1,3":6,
  "4,2,1":5, "4,2,2":6, "4,2,3":7,
  "4,3,1":6, "4,3,2":7, "4,3,3":8
};
// Table B最終スコア = min((U + F + W) - 2, 5) + Coupling補正
function getScoreB(upperArmScore, forearmScore, wristScore, coupling) {
  let base = (upperArmScore + forearmScore + wristScore) - 2;
  if (base > 5) base = 5;
  return base + coupling;
}

// Table C: 組み合わせ表 (ScoreA, ScoreB) → 統合スコア (1～12)
const tableC_Lookup = {
  "1,1":1, "1,2":2, "1,3":3, "1,4":4, "1,5":5,
  "2,1":2, "2,2":3, "2,3":4, "2,4":5, "2,5":6,
  "3,1":3, "3,2":4, "3,3":5, "3,4":6, "3,5":7,
  "4,1":4, "4,2":5, "4,3":6, "4,4":7, "4,5":8,
  "5,1":5, "5,2":6, "5,3":7, "5,4":8, "5,5":9,
  "6,1":6, "6,2":7, "6,3":8, "6,4":9, "6,5":10,
  "7,1":7, "7,2":8, "7,3":9, "7,4":10, "7,5":11,
  "8,1":8, "8,2":9, "8,3":10, "8,4":11, "8,5":12
};
function getTableCScore(scoreA, scoreB) {
  const key = `${scoreA},${scoreB}`;
  return tableC_Lookup[key] || 1;
}

// ===== 各部位の評価関数 =====

// 体幹：calcTrunkScore(体幹屈曲角, 回旋/側屈ありか)
// 外部入力の trunkLateralFlexion（0,5,10）があれば側屈ありと判定
function evaluateTrunk(computedAngles, calibInputs) {
  let rotationFlag = Math.abs(computedAngles.trunkRotationAngle) >= 10; // 10°以上ならあり
  let sideBendFlag = calibInputs.trunkLateralFlexion > 0;
  return calcTrunkScore(computedAngles.trunkFlexionAngle, rotationFlag, sideBendFlag);
}

// 頸：calcNeckScore(頸角, 回旋/側屈ありか)
function evaluateNeck(computedAngles, calibInputs) {
  let rotationFlag = calibInputs.neckRotation > 0;
  let sideBendFlag = calibInputs.neckLateralBending > 0;
  return calcNeckScore(computedAngles.neckAngle, rotationFlag, sideBendFlag);
}

// 下肢：calcLegScore(weightBearing, 膝角度, postureType)
function evaluateLeg(computedAngles, calibInputs) {
  let avgKnee = (computedAngles.leftKneeAngle + computedAngles.rightKneeAngle) / 2;
  return calcLegScore(calibInputs.weightBearing, avgKnee, calibInputs.postureType);
}

// 上肢（上腕）：calcUpperArmScore(肩角, 外転/回旋, 肩挙上, 重力補助)
// 平均値を使用
function evaluateUpperArm(computedAngles, calibInputs) {
  let avg = (computedAngles.leftShoulderAngle + computedAngles.rightShoulderAngle) / 2;
  return calcUpperArmScore(avg, calibInputs.upperArmCorrection, calibInputs.shoulderElevation, calibInputs.gravityAssist);
}

// 前腕：calcForearmScore(肘角度)
function evaluateForearm(computedAngles) {
  let avg = (computedAngles.leftElbowAngle + computedAngles.rightElbowAngle) / 2;
  return calcForearmScore(avg);
}

// 手首：calcWristScore(手首角度, 手首補正)
function evaluateWrist(computedAngles, calibInputs) {
  let avg = (computedAngles.leftWristAngle + computedAngles.rightWristAngle) / 2;
  return calcWristScore(avg, calibInputs.wristCorrection);
}

// 活動度：各項目（静的姿勢, 反復動作, 不安定動作）の合計 (0～3)
function evaluateActivity(calibInputs) {
  return calibInputs.staticPosture + calibInputs.repetitiveMovement + calibInputs.unstableMovement;
}

// ===== 最終REBAスコアの算出 =====
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

// ===== リスク判定 =====
function getRiskLevel(score) {
  if (score === 1) return "無視できる";
  else if (score >= 2 && score <= 3) return "低リスク";
  else if (score >= 4 && score <= 7) return "中リスク";
  else if (score >= 8 && score <= 10) return "高リスク";
  else return "非常に高リスク";
}

// ===== 結果表示 =====
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

// ===== Webカメラ映像からの検出ループ =====
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
    canvasCtx.save();
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
    canvasCtx.restore();
  });
  
  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}