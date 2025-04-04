// Copyright 2023 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";
const demosSection = document.getElementById("demos");
let poseLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoHeight = "360px";
const videoWidth = "480px";
// Before we can use PoseLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
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
/********************************************************************
// Demo 1: Grab a bunch of images from the page and detection them
// upon click.
********************************************************************/
// In this demo, we have put all our clickable images in divs with the
// CSS class 'detectionOnClick'. Lets get all the elements that have
// this class.
const imageContainers = document.getElementsByClassName("detectOnClick");
// Now let's go through all of these and add a click event listener.
for (let i = 0; i < imageContainers.length; i++) {
    // Add event listener to the child element whichis the img element.
    imageContainers[i].children[0].addEventListener("click", handleClick);
}
// When an image is clicked, let's detect it and display results!
async function handleClick(event) {
    if (!poseLandmarker) {
        console.log("Wait for poseLandmarker to load before clicking!");
        return;
    }
    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        await poseLandmarker.setOptions({ runningMode: "IMAGE" });
    }
    // Remove all landmarks drawed before
    const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
    for (var i = allCanvas.length - 1; i >= 0; i--) {
        const n = allCanvas[i];
        n.parentNode.removeChild(n);
    }
    // We can call poseLandmarker.detect as many times as we like with
    // different image data each time. The result is returned in a callback.
    poseLandmarker.detect(event.target, (result) => {
        const canvas = document.createElement("canvas");
        canvas.setAttribute("class", "canvas");
        canvas.setAttribute("width", event.target.naturalWidth + "px");
        canvas.setAttribute("height", event.target.naturalHeight + "px");
        canvas.style =
            "left: 0px;" +
                "top: 0px;" +
                "width: " +
                event.target.width +
                "px;" +
                "height: " +
                event.target.height +
                "px;";
        event.target.parentNode.appendChild(canvas);
        const canvasCtx = canvas.getContext("2d");
        const drawingUtils = new DrawingUtils(canvasCtx);
        for (const landmark of result.landmarks) {
            drawingUtils.drawLandmarks(landmark, {
                radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
            });
            drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
        }
    });
}
/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);
// Check if webcam access is supported.
const hasGetUserMedia = () => { var _a; return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia); };
// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
}
else {
    console.warn("getUserMedia() is not supported by your browser");
}
// Enable the live webcam view and start detection.
function enableCam(event) {
    if (!poseLandmarker) {
        console.log("Wait! poseLandmaker not loaded yet.");
        return;
    }
    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "Recording Start";
    }
    else {
        webcamRunning = true;
        enableWebcamButton.innerText = "Recording Stop";
    }
    // getUsermedia parameters.
    const constraints = {
        video: true
    };
    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}
let lastVideoTime = -1;

// --- 角度計算用ユーティリティ関数 ---

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
 * ベクトルと垂直（(0,-1)）との角度を算出（度数に変換）
 * ※画像上では y 座標が下方向に増加するため、垂直は (0, -1) と定義
 * @param {object} vector {x, y}
 * @returns {number} 角度（度）
 */
function angleWithVertical(vector) {
  const vertical = { x: 0, y: -1 };
  const dot = vector.x * vertical.x + vector.y * vertical.y; // = -vector.y
  const normVector = Math.sqrt(vector.x * vector.x + vector.y * vector.y);
  const angleRad = Math.acos(dot / normVector);
  return angleRad * (180 / Math.PI);
}

// --- REBA用関節角度算出関数 ---
// 検出された１ポーズのlandmarksを受け取り、各部位の角度を計算して返す例です。
function computeREBAAngles(landmarks) {

  const requiredIndexes = [0, 11, 12, 13, 15, 19, 23, 24, 25, 27, 14, 16, 20, 26, 28];
  // 必要なランドマークがすべて存在するか確認
  for (const index of requiredIndexes) {
    if (!landmarks[index]) {
      console.warn(`landmark ${index} is missing.`);
      return null;
    }
  }
  
  // ランドマークの対応（MediaPipe Poseのインデックス）
  // 0: nose, 11: left shoulder, 12: right shoulder,
  // 13: left elbow, 14: right elbow, 15: left wrist, 16: right wrist,
  // 19: left index, 20: right index, 23: left hip, 24: right hip,
  // 25: left knee, 26: right knee, 27: left ankle, 28: right ankle

  // 1. トランク角度（両肩中点と両股関節中点から）
  const shoulderCenter = {
    x: (landmarks[11].x + landmarks[12].x) / 2,
    y: (landmarks[11].y + landmarks[12].y) / 2
  };
  const hipCenter = {
    x: (landmarks[23].x + landmarks[24].x) / 2,
    y: (landmarks[23].y + landmarks[24].y) / 2
  };
  const trunkVector = {
    x: shoulderCenter.x - hipCenter.x,
    y: shoulderCenter.y - hipCenter.y
  };
  const trunkAngle = angleWithVertical(trunkVector);

  // 2. 首角度（鼻と両肩中点から）
  const nose = landmarks[0];
  const neckVector = {
    x: nose.x - shoulderCenter.x,
    y: nose.y - shoulderCenter.y
  };
  const neckAngle = angleWithVertical(neckVector);

  // 3. 左上腕角度（左肩から左肘のベクトルと垂直との角度）
  const leftUpperArmVector = {
    x: landmarks[13].x - landmarks[11].x,
    y: landmarks[13].y - landmarks[11].y
  };
  const leftUpperArmAngle = angleWithVertical(leftUpperArmVector);

  // 4. 左下腕角度（左肩, 左肘, 左手首）
  const leftLowerArmAngle = calculateAngle(landmarks[11], landmarks[13], landmarks[15]);

  // 5. 左手首角度（左肘, 左手首, 左人差し指）
  const leftWristAngle = calculateAngle(landmarks[13], landmarks[15], landmarks[19]);

  // 6. 右上腕角度（右肩から右肘のベクトルと垂直との角度）
  const rightUpperArmVector = {
    x: landmarks[14].x - landmarks[12].x,
    y: landmarks[14].y - landmarks[12].y
  };
  const rightUpperArmAngle = angleWithVertical(rightUpperArmVector);

  // 7. 右下腕角度（右肩, 右肘, 右手首）
  const rightLowerArmAngle = calculateAngle(landmarks[12], landmarks[14], landmarks[16]);

  // 8. 右手首角度（右肘, 右手首, 右人差し指）
  const rightWristAngle = calculateAngle(landmarks[14], landmarks[16], landmarks[20]);

  // 9. 左膝角度（左股関節, 左膝, 左足首）
  const leftKneeAngle = calculateAngle(landmarks[23], landmarks[25], landmarks[27]);

  // 10. 右膝角度（右股関節, 右膝, 右足首）
  const rightKneeAngle = calculateAngle(landmarks[24], landmarks[26], landmarks[28]);

  // 結果をオブジェクトにまとめる
  return {
    trunkAngle,
    neckAngle,
    leftUpperArmAngle,
    leftLowerArmAngle,
    leftWristAngle,
    rightUpperArmAngle,
    rightLowerArmAngle,
    rightWristAngle,
    leftKneeAngle,
    rightKneeAngle
  };
}

function displayAngles(ctx, angles) {
  ctx.fillStyle = "red";
  ctx.font = "16px Arial";
  const lineHeight = 18;
  const startX = 10;
  let startY = 20;
  const lines = [
    `Trunk Angle: ${angles.trunkAngle.toFixed(1)}°`,
    `Neck Angle: ${angles.neckAngle.toFixed(1)}°`,
    `Left Upper Arm: ${angles.leftUpperArmAngle.toFixed(1)}°`,
    `Left Lower Arm: ${angles.leftLowerArmAngle.toFixed(1)}°`,
    `Left Wrist: ${angles.leftWristAngle.toFixed(1)}°`,
    `Right Upper Arm: ${angles.rightUpperArmAngle.toFixed(1)}°`,
    `Right Lower Arm: ${angles.rightLowerArmAngle.toFixed(1)}°`,
    `Right Wrist: ${angles.rightWristAngle.toFixed(1)}°`,
    `Left Knee: ${angles.leftKneeAngle.toFixed(1)}°`,
    `Right Knee: ${angles.rightKneeAngle.toFixed(1)}°`
  ];
  lines.forEach(line => {
    ctx.fillText(line, startX, startY);
    startY += lineHeight;
  });
}


// --- 例: Webカメラからの検出結果内での利用 ---
// predictWebcam() 内の poseLandmarker.detectForVideo() のコールバック内で
// 各ポーズについて角度を計算し、結果をコンソールに出力します。

async function predictWebcam() {
  // ※既存のキャンバス描画等のコードはそのまま残す
  canvasElement.style.height = videoHeight;
  video.style.height = videoHeight;
  canvasElement.style.width = videoWidth;
  video.style.width = videoWidth;

  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await poseLandmarker.setOptions({ runningMode: "VIDEO" });
  }
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    // predictWebcam() のコールバック内での利用例
    poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      
      for (const landmarkSet of result.landmarks) {
        drawingUtils.drawLandmarks(landmarkSet, {
          radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
        });
        drawingUtils.drawConnectors(landmarkSet, PoseLandmarker.POSE_CONNECTIONS);
    
        const angles = computeREBAAngles(landmarkSet);
        if (angles) {
          console.log("REBA angles:", angles);
          displayAngles(canvasCtx, angles);
        } else {
          console.warn("角度計算に必要なランドマークが不足しています。");
        }
      }
      canvasCtx.restore();
    });
  }
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}
