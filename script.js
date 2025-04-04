import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

let poseLandmarker;
let runningMode = "IMAGE";
let webcamRunning = false;
let maxRebaScore = 0;
let maxRiskLevel = "";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

const createPoseLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
      delegate: "GPU"
    },
    runningMode,
    numPoses: 1
  });
};
createPoseLandmarker();

function enableCam() {
  webcamRunning = !webcamRunning;
  document.getElementById("webcamButton").innerText = webcamRunning ? "Stop Recording" : "Start Recording";

  if (webcamRunning) {
    navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } }
    }).then((stream) => {
      video.srcObject = stream;
      video.play();
      video.addEventListener("loadeddata", () => {
        predictWebcam();
      });
    });
  } else {
    alert(`Max REBA Score: ${maxRebaScore}
Risk Level: ${maxRiskLevel}`);
  }
}

document.getElementById("webcamButton").addEventListener("click", enableCam);

function getAngleAndCrossProduct(a, b, c) {
  if (!a || !b || !c) throw new Error("関節データが欠損しています");
  const vecA = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const vecB = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const dot = vecA.x * vecB.x + vecA.y * vecB.y + vecA.z * vecB.z;
  const magA = Math.hypot(vecA.x, vecA.y, vecA.z);
  const magB = Math.hypot(vecB.x, vecB.y, vecB.z);
  const angle = Math.acos(dot / (magA * magB)) * (180 / Math.PI);
  const cross = {
    x: vecA.y * vecB.z - vecA.z * vecB.y,
    y: vecA.z * vecB.x - vecA.x * vecB.z,
    z: vecA.x * vecB.y - vecA.y * vecB.x,
  };
  return { angle, cross };
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
}

function getRebaAngles(lm) {
  const safe = (a, b, c) => getAngleAndCrossProduct(a, b, c).angle;
  const leg = Math.max(safe(lm[23], lm[25], lm[27]), safe(lm[24], lm[26], lm[28]));
  const upperArm = Math.max(safe(lm[23], lm[11], lm[13]), safe(lm[24], lm[12], lm[14]));
  const lowerArm = Math.max(safe(lm[11], lm[13], lm[15]), safe(lm[12], lm[14], lm[16]));
  const wrist = Math.max(safe(lm[13], lm[15], lm[19]), safe(lm[14], lm[16], lm[20]));
  return {
    neck: safe(midpoint(lm[11], lm[12]), lm[0], { x: 0, y: -1, z: 0 }),
    trunk: safe(midpoint(lm[23], lm[24]), midpoint(lm[11], lm[12]), { x: 0, y: -1, z: 0 }),
    leg, upperArm, lowerArm, wrist
  };
}

function getModifiersFromUI() {
  const load = document.querySelector('input[name="loadScore"]:checked')?.value || "0";
  const coupling = document.querySelector('input[name="couplingScore"]:checked')?.value || "0";
  const activity = document.querySelector('input[name="activityScore"]:checked')?.value || "0";
  const kneeMode = document.querySelector('input[name="kneeMode"]:checked')?.value || "both";
  return { load, coupling, activity, kneeMode };
}

function calculateREBAScore(angles, modifiers) {
  let neckScore = (angles.neck > 20) ? 2 : 1;
  let trunkScore = (angles.trunk > 60) ? 3 : (angles.trunk > 20 ? 2 : 1);
  let legScore = 1;
  if (modifiers.kneeMode === "sitwalk") legScore = 1;
  else if (modifiers.kneeMode === "one") legScore = 2;
  else if (angles.leg > 30) legScore = 2;

  const scoreA = trunkScore + neckScore + legScore + parseInt(modifiers.load);
  const upperArmScore = (angles.upperArm > 90) ? 3 : (angles.upperArm > 45 ? 2 : 1);
  const lowerArmScore = (angles.lowerArm < 60) ? 2 : 1;
  const wristScore = (angles.wrist > 15) ? 2 : 1;
  const scoreB = upperArmScore + lowerArmScore + wristScore + parseInt(modifiers.coupling);

  const tableC = [
    [1,2,3,4,5,6], [2,3,4,5,6,7], [3,4,5,6,7,8],
    [4,5,6,7,8,9], [5,6,7,8,9,10], [6,7,8,9,10,11],
    [7,8,9,10,11,12], [8,9,10,11,12,13], [9,10,11,12,13,14],
  ];
  const aIdx = Math.min(scoreA - 1, 8);
  const bIdx = Math.min(scoreB - 1, 5);
  const scoreC = tableC[aIdx][bIdx];
  const finalScore = Math.min(scoreC + parseInt(modifiers.activity), 15);
  let riskLevel = "";
  if (finalScore <= 1) riskLevel = "0: 無視できる";
  else if (finalScore <= 3) riskLevel = "1: 低リスク（必要に応じて）";
  else if (finalScore <= 7) riskLevel = "2: 中リスク（対応必要）";
  else if (finalScore <= 10) riskLevel = "3: 高リスク（早急に対応）";
  else riskLevel = "4: 非常に高リスク（即時対応）";
  return { scoreA, scoreB, scoreC, finalScore, riskLevel };
}

let lastVideoTime = -1;
let lastAngleTime = 0;

async function predictWebcam() {
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await poseLandmarker.setOptions({ runningMode: "VIDEO" });
  }

  const startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;

    poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
      const lm = result.landmarks?.[0];
      if (!lm) return;
      const now = performance.now();
      if (now - lastAngleTime > 500) {
        lastAngleTime = now;
        try {
          const modifiers = getModifiersFromUI();
          const angles = getRebaAngles(lm);
          const scores = calculateREBAScore(angles, modifiers);
          if (scores.finalScore > maxRebaScore) {
            maxRebaScore = scores.finalScore;
            maxRiskLevel = scores.riskLevel;
          }
          updateRebaChart(scores.finalScore);
        } catch (e) {
          console.warn("計算エラー:", e.message);
        }
      }

      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);
      if (result.landmarks) {
        for (const landmarks of result.landmarks) {
          drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
            color: "#00FF00", lineWidth: 4
          });
          drawingUtils.drawLandmarks(landmarks, {
            color: "#FF6F00", radius: 3
          });
        }
      }
      canvasCtx.restore();
    });
  }

  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}


// Chart.js 初期化
const ctx = document.getElementById('rebaChart').getContext('2d');
const rebaChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      {
        label: 'Score A',
        data: [],
        borderColor: 'rgba(0, 123, 255, 1)',
        backgroundColor: 'rgba(0, 123, 255, 0.1)',
        borderWidth: 2,
        fill: false,
        tension: 0.1
      },
      {
        label: 'Score B',
        data: [],
        borderColor: 'rgba(40, 167, 69, 1)',
        backgroundColor: 'rgba(40, 167, 69, 0.1)',
        borderWidth: 2,
        fill: false,
        tension: 0.1
      },
      {
        label: 'Score C',
        data: [],
        borderColor: 'rgba(255, 193, 7, 1)',
        backgroundColor: 'rgba(255, 193, 7, 0.1)',
        borderWidth: 2,
        fill: false,
        tension: 0.1
      },
      {
        label: 'REBA Score',
        data: [],
        borderColor: 'rgba(220, 53, 69, 1)',
        backgroundColor: 'rgba(220, 53, 69, 0.1)',
        borderWidth: 2,
        fill: false,
        tension: 0.1
      }
    ]
  },
  options: {
    responsive: true,
    animation: false,
    scales: {
      y: {
        suggestedMin: 0,
        suggestedMax: 15
      }
    }
  }
});


function updateRebaChart(scores) {
  const now = new Date();
  const time = now.toLocaleTimeString();
  const data = rebaChart.data;

  data.labels.push(time);
  data.datasets[0].data.push(scores.scoreA);
  data.datasets[1].data.push(scores.scoreB);
  data.datasets[2].data.push(scores.scoreC);
  data.datasets[3].data.push(scores.finalScore);

  if (data.labels.length > 30) {
    data.labels.shift();
    data.datasets.forEach(ds => ds.data.shift());
  }

  rebaChart.update();
}
