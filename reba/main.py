# main.py (完全版)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Dict, Any
import math
from fastapi.middleware.cors import CORSMiddleware # CORS用

app = FastAPI(title="REBA Evaluation API")

# -----------------------------
# モデル定義 (Pydantic)
# -----------------------------
class Landmark(BaseModel):
    # ↓↓↓ Class definition body MUST be indented ↓↓↓
    x: float
    y: float
    z: float = 0.0
    visibility: float = Field(default=1.0, ge=0.0, le=1.0)

class CalibrationInputs(BaseModel):
    # ↓↓↓ Indentation required ↓↓↓
    filmingSide: str = Field(..., description="left or right")
    neckRotation: float # Treat as flag: > 0 means rotated
    neckLateralBending: float # Treat as flag: > 0 means bent
    trunkLateralFlexion: float # Treat as flag: > 0 means bent
    loadForce: float # Represents score bracket start (0, 5, 11)
    postureCategory: str # "standingBoth", "standingOne", "sittingWalking"
    upperArmCorrection: float # Abduction/Rotation flag (0 or 1)
    shoulderElevation: float # Shoulder raised flag (0 or 1)
    gravityAssist: float # Arm supported flag (0 or -1)
    wristCorrection: float # Deviation/Twist flag (0 or 1)
    staticPosture: int # Activity score flag (0 or 1)
    repetitiveMovement: int # Activity score flag (0 or 1)
    unstableMovement: int # Activity score flag (0 or 1)
    coupling: int # Coupling score addition (0, 1, 2, or 3)

    @validator('coupling')
    def coupling_must_be_valid(cls, v):
        # Allow 0, 1, 2, 3 based on HTML radio buttons
        if not 0 <= v <= 3:
            raise ValueError('Coupling score must be between 0 and 3')
        return v

    @validator('filmingSide')
    def side_must_be_valid(cls, v):
        if v not in ["left", "right"]:
             raise ValueError('Filming side must be "left" or "right"')
        return v

    @validator('postureCategory')
    def posture_must_be_valid(cls, v):
        if v not in ["standingBoth", "standingOne", "sittingWalking"]:
             raise ValueError('Invalid posture category')
        return v

    # Add other validators if needed (e.g., ensuring flags are 0/1)

class REBAInput(BaseModel):
    # ↓↓↓ Indentation required ↓↓↓
    landmarks: List[Landmark]
    calibInputs: CalibrationInputs

# -----------------------------
# 角度計算ユーティリティ
# -----------------------------
def calculate_angle(p1: Dict[str, float], p2: Dict[str, float], p3: Dict[str, float], min_visibility: float = 0.5) -> float:
    """ Calculates the angle between three points (p1, p2, p3), where p2 is the vertex. """
    # ↓↓↓ Function body MUST be indented ↓↓↓
    if p1.get('visibility', 1.0) < min_visibility or \
       p2.get('visibility', 1.0) < min_visibility or \
       p3.get('visibility', 1.0) < min_visibility:
        return 0.0

    v1 = {"x": p1["x"] - p2["x"], "y": p1["y"] - p2["y"]}
    v2 = {"x": p3["x"] - p2["x"], "y": p3["y"] - p2["y"]}
    dot = v1["x"] * v2["x"] + v1["y"] * v2["y"]
    mag1 = math.sqrt(v1["x"]**2 + v1["y"]**2)
    mag2 = math.sqrt(v2["x"]**2 + v2["y"]**2)

    if mag1 * mag2 == 0: return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)

def angle_with_vertical(p1: Dict[str, float], p2: Dict[str, float], min_visibility: float = 0.5) -> float:
    """ Calculates the angle of the vector p1 -> p2 with the downward vertical. """
    # ↓↓↓ Indentation required ↓↓↓
    if p1.get('visibility', 1.0) < min_visibility or p2.get('visibility', 1.0) < min_visibility:
        return 0.0

    vector = {"x": p2["x"] - p1["x"], "y": p2["y"] - p1["y"]}
    vertical = {"x": 0, "y": 1} # Y increases downwards
    dot = vector["x"] * vertical["x"] + vector["y"] * vertical["y"]
    norm = math.sqrt(vector["x"]**2 + vector["y"]**2)

    if norm == 0: return 0.0
    cos_angle = max(-1.0, min(1.0, dot / norm))
    angle_rad = math.acos(cos_angle)
    # Returns angle where 0=down, 90=horizontal, 180=up
    # Adjust if REBA expects 0=vertical/aligned with trunk
    return math.degrees(angle_rad)

# -----------------------------
# ランドマークから角度計算
# -----------------------------
def compute_all_angles(landmarks: List[Landmark]) -> Dict[str, float]:
    # ↓↓↓ Indentation required ↓↓↓
    lm_indices = { "Nose": 0, "L_Shoulder": 11, "R_Shoulder": 12, "L_Elbow": 13, "R_Elbow": 14, "L_Wrist": 15, "R_Wrist": 16, "L_Hip": 23, "R_Hip": 24, "L_Knee": 25, "R_Knee": 26, "L_Ankle": 27, "R_Ankle": 28 }
    max_index = max(lm_indices.values())
    if len(landmarks) <= max_index:
         raise HTTPException(status_code=400, detail=f"Not enough landmarks provided ({len(landmarks)}). Need at least {max_index + 1}")

    # Use model_dump() for Pydantic v2, dict() for v1
    lm = [landmark.model_dump() if hasattr(landmark, 'model_dump') else landmark.dict() for landmark in landmarks]

    # Midpoints
    shoulder_mid = {"x": (lm[lm_indices["L_Shoulder"]]["x"] + lm[lm_indices["R_Shoulder"]]["x"]) / 2, "y": (lm[lm_indices["L_Shoulder"]]["y"] + lm[lm_indices["R_Shoulder"]]["y"]) / 2, "visibility": min(lm[lm_indices["L_Shoulder"]].get('visibility',1.0), lm[lm_indices["R_Shoulder"]].get('visibility',1.0))}
    hip_mid = {"x": (lm[lm_indices["L_Hip"]]["x"] + lm[lm_indices["R_Hip"]]["x"]) / 2, "y": (lm[lm_indices["L_Hip"]]["y"] + lm[lm_indices["R_Hip"]]["y"]) / 2, "visibility": min(lm[lm_indices["L_Hip"]].get('visibility',1.0), lm[lm_indices["R_Hip"]].get('visibility',1.0))}

    # Neck Angle (Flexion/Extension relative to vertical)
    neckAngle = angle_with_vertical(shoulder_mid, lm[lm_indices["Nose"]])

    # Trunk Angle (Flexion/Extension relative to vertical)
    trunkFlexionAngle = angle_with_vertical(hip_mid, shoulder_mid)

    # Trunk Rotation (Approximate)
    shoulder_dx = lm[lm_indices["R_Shoulder"]]["x"] - lm[lm_indices["L_Shoulder"]]["x"]
    shoulder_dy = lm[lm_indices["R_Shoulder"]]["y"] - lm[lm_indices["L_Shoulder"]]["y"]
    hip_dx = lm[lm_indices["R_Hip"]]["x"] - lm[lm_indices["L_Hip"]]["x"]
    hip_dy = lm[lm_indices["R_Hip"]]["y"] - lm[lm_indices["L_Hip"]]["y"]
    shoulder_angle_rad = math.atan2(shoulder_dy, shoulder_dx) if shoulder_dx != 0 or shoulder_dy != 0 else 0
    hip_angle_rad = math.atan2(hip_dy, hip_dx) if hip_dx != 0 or hip_dy != 0 else 0
    trunkRotationAngle = math.degrees(shoulder_angle_rad - hip_angle_rad)
    trunkRotationAngle = (trunkRotationAngle + 180) % 360 - 180

    # Limb Angles (Internal)
    leftUpperArmAngle = calculate_angle(lm[lm_indices["L_Hip"]], lm[lm_indices["L_Shoulder"]], lm[lm_indices["L_Elbow"]])
    rightUpperArmAngle = calculate_angle(lm[lm_indices["R_Hip"]], lm[lm_indices["R_Shoulder"]], lm[lm_indices["R_Elbow"]])
    leftElbowAngle = calculate_angle(lm[lm_indices["L_Shoulder"]], lm[lm_indices["L_Elbow"]], lm[lm_indices["L_Wrist"]])
    rightElbowAngle = calculate_angle(lm[lm_indices["R_Shoulder"]], lm[lm_indices["R_Elbow"]], lm[lm_indices["R_Wrist"]])
    # Wrist angle calculation is unreliable from these landmarks alone for REBA's flexion/extension need
    leftWristAngle = 0.0
    rightWristAngle = 0.0
    leftKneeAngle = calculate_angle(lm[lm_indices["L_Hip"]], lm[lm_indices["L_Knee"]], lm[lm_indices["L_Ankle"]])
    rightKneeAngle = calculate_angle(lm[lm_indices["R_Hip"]], lm[lm_indices["R_Knee"]], lm[lm_indices["R_Ankle"]])

    return {
        "neckAngle": neckAngle, "trunkFlexionAngle": trunkFlexionAngle, "trunkRotationAngle": trunkRotationAngle,
        "leftUpperArmAngle": leftUpperArmAngle, "rightUpperArmAngle": rightUpperArmAngle,
        "leftElbowAngle": leftElbowAngle, "rightElbowAngle": rightElbowAngle,
        "leftWristAngle": leftWristAngle, "rightWristAngle": rightWristAngle,
        "leftKneeAngle": leftKneeAngle, "rightKneeAngle": rightKneeAngle
    }

# -----------------------------
# REBAスコア計算関数群
# (Need careful review against official REBA scoring sheet for accuracy)
# -----------------------------
def calc_neck_score(neckAngle: float, rotationFlag: bool, sideBendFlag: bool) -> int:
    base = 1 if neckAngle <= 20 else 2 # Simplified: Assumes positive angle = flexion, 0 = upright
    add = (1 if rotationFlag else 0) + (1 if sideBendFlag else 0)
    return base + add

def calc_trunk_score(trunkFlexAngle: float, rotationFlag: bool, sideBendFlag: bool) -> int:
    # Simplified: Assumes positive angle = flexion, 0 = upright
    if trunkFlexAngle <= 5: base = 1
    elif trunkFlexAngle <= 20: base = 2
    elif trunkFlexAngle <= 60: base = 3
    else: base = 4
    add = (1 if rotationFlag else 0) + (1 if sideBendFlag else 0)
    return base + add

def calc_leg_score_unified(postureCategory: str, kneeFlexAngle: float) -> int:
    if postureCategory == "sittingWalking": return 1 # Sitting
    base = 1 if postureCategory == "standingBoth" else 2 # Standing
    # Assuming kneeFlexAngle is internal angle (0=straight). REBA uses flexion.
    flex = 180 - kneeFlexAngle
    add = 0
    if 30 <= flex <= 60: add = 1
    elif flex > 60: add = 2
    return base + add

def calc_upper_arm_score(upperArmAngle: float, upperArmCorrection: float, shoulderElevation: float, gravityAssist: float) -> int:
    # Using Hip-Shoulder-Elbow angle as proxy, REBA definition differs. Requires adjustment.
    # Rough approximation mapping:
    angle_proxy = abs(upperArmAngle - 90) # Make 0 roughly vertical
    if angle_proxy <= 20: base = 1
    elif angle_proxy <= 45: base = 2
    elif angle_proxy <= 90: base = 3
    else: base = 4
    # Corrections: +1 Abduction/Rotation, +1 Shoulder Raised, -1 Supported
    corrected = base + int(upperArmCorrection) + int(shoulderElevation) + int(gravityAssist)
    return max(1, min(6, corrected)) # Clamp to REBA range 1-6

def calc_forearm_score(elbowAngle: float) -> int:
    # Assumes elbowAngle is internal angle. REBA 60-100 = 1, else 2
    return 1 if 60 <= elbowAngle <= 100 else 2

def calc_wrist_score(wristAngle: float, wristCorrection: float) -> int:
    # Calculated wristAngle is unreliable (set to 0.0). Base score relies on assumption or input.
    # Assume base 1 (neutral) unless user input suggests otherwise via wristCorrection?
    # REBA: 0-15 deg flexion = 1, >15 deg = 2. +1 for deviation/twist.
    base = 1 # Defaulting to base 1 as calc is unreliable
    score = base + int(wristCorrection)
    return max(1, min(3, score)) # REBA wrist score range 1-3? Verify.

def calc_load_score(loadKgInput: float) -> int:
    # Input value represents start of bracket (0, 5, 11)
    if loadKgInput < 5: return 0 # < 5kg
    elif loadKgInput <= 10: return 1 # 5-10kg
    else: return 2 # > 10kg

# -----------------------------
# ルックアップテーブル & ヘルパー
# -----------------------------
# NOTE: Ensure keys match the max possible scores from calc functions
tableA_Lookup = { "1,1,1":1, "1,1,2":2, "1,1,3":3, "1,1,4":4, "1,2,1":1, "1,2,2":2, "1,2,3":3, "1,2,4":4, "1,3,1":3, "1,3,2":3, "1,3,3":5, "1,3,4":6, "1,4,1":4,"1,4,2":4,"1,4,3":6,"1,4,4":7, "2,1,1":2, "2,1,2":3, "2,1,3":4, "2,1,4":5, "2,2,1":3, "2,2,2":4, "2,2,3":5, "2,2,4":6, "2,3,1":4, "2,3,2":5, "2,3,3":6, "2,3,4":7, "2,4,1":5,"2,4,2":6,"2,4,3":7,"2,4,4":8, "3,1,1":2, "3,1,2":4, "3,1,3":5, "3,1,4":6, "3,2,1":4, "3,2,2":5, "3,2,3":6, "3,2,4":7, "3,3,1":5, "3,3,2":6, "3,3,3":7, "3,3,4":8, "3,4,1":6,"3,4,2":7,"3,4,3":8,"3,4,4":9, "4,1,1":3, "4,1,2":5, "4,1,3":6, "4,1,4":7, "4,2,1":5, "4,2,2":6, "4,2,3":7, "4,2,4":8, "4,3,1":6, "4,3,2":7, "4,3,3":8, "4,3,4":9, "4,4,1":7,"4,4,2":8,"4,4,3":9,"4,4,4":9, "5,1,1":4,"5,1,2":6,"5,1,3":7,"5,1,4":8,"5,2,1":6,"5,2,2":7,"5,2,3":8,"5,2,4":9,"5,3,1":7,"5,3,2":8,"5,3,3":9,"5,3,4":9,"5,4,1":8,"5,4,2":9,"5,4,3":9,"5,4,4":9, "6,1,1":5,"6,1,2":7,"6,1,3":8,"6,1,4":9,"6,2,1":7,"6,2,2":8,"6,2,3":9,"6,2,4":9,"6,3,1":8,"6,3,2":9,"6,3,3":9,"6,3,4":9,"6,4,1":9,"6,4,2":9,"6,4,3":9,"6,4,4":9 }
tableB_Lookup = { "1,1,1":1,"1,1,2":1,"1,1,3":2,"1,2,1":2,"1,2,2":2,"1,2,3":3,"2,1,1":2,"2,1,2":3,"2,1,3":3,"2,2,1":3,"2,2,2":4,"2,2,3":5,"3,1,1":3,"3,1,2":4,"3,1,3":4,"3,2,1":4,"3,2,2":5,"3,2,3":6,"4,1,1":4,"4,1,2":5,"4,1,3":5,"4,2,1":5,"4,2,2":6,"4,2,3":7,"5,1,1":5,"5,1,2":6,"5,1,3":6,"5,2,1":6,"5,2,2":7,"5,2,3":8,"6,1,1":6,"6,1,2":7,"6,1,3":7,"6,2,1":7,"6,2,2":8,"6,2,3":9 }
tableC_Lookup = { "1,1":1,"1,2":1,"1,3":1,"1,4":2,"1,5":3,"1,6":3,"1,7":4,"1,8":5,"1,9":6,"1,10":7,"1,11":7,"1,12":7,"2,1":1,"2,2":2,"2,3":2,"2,4":3,"2,5":4,"2,6":4,"2,7":5,"2,8":6,"2,9":6,"2,10":7,"2,11":7,"2,12":8,"3,1":2,"3,2":3,"3,3":3,"3,4":4,"3,5":5,"3,6":6,"3,7":7,"3,8":7,"3,9":7,"3,10":8,"3,11":8,"3,12":8,"4,1":3,"4,2":4,"4,3":4,"4,4":4,"4,5":5,"4,6":6,"4,7":7,"4,8":8,"4,9":8,"4,10":9,"4,11":9,"4,12":9,"5,1":4,"5,2":4,"5,3":4,"5,4":5,"5,5":6,"5,6":7,"5,7":8,"5,8":8,"5,9":9,"5,10":9,"5,11":9,"5,12":9,"6,1":6,"6,2":6,"6,3":6,"6,4":7,"6,5":8,"6,6":8,"6,7":9,"6,8":9,"6,9":10,"6,10":10,"6,11":10,"6,12":10,"7,1":7,"7,2":7,"7,3":7,"7,4":8,"7,5":9,"7,6":9,"7,7":9,"7,8":10,"7,9":10,"7,10":11,"7,11":11,"7,12":11,"8,1":8,"8,2":8,"8,3":8,"8,4":9,"8,5":10,"8,6":10,"8,7":10,"8,8":10,"8,9":11,"8,10":11,"8,11":11,"8,12":12,"9,1":9,"9,2":9,"9,3":9,"9,4":10,"9,5":10,"9,6":11,"9,7":11,"9,8":11,"9,9":12,"9,10":12,"9,11":12,"9,12":12,"10,1":10,"10,2":10,"10,3":10,"10,4":11,"10,5":11,"10,6":11,"10,7":12,"10,8":12,"10,9":12,"10,10":12,"10,11":12,"10,12":12,"11,1":11,"11,2":11,"11,3":11,"11,4":12,"11,5":12,"11,6":12,"11,7":12,"11,8":12,"11,9":12,"11,10":12,"11,11":12,"11,12":12,"12,1":12,"12,2":12,"12,3":12,"12,4":12,"12,5":12,"12,6":12,"12,7":12,"12,8":12,"12,9":12,"12,10":12,"12,11":12,"12,12":12 }

def lookup_score(table: Dict[str, int], key_parts: List[int], min_vals: List[int], max_vals: List[int]) -> int:
    # ↓↓↓ Indentation required ↓↓↓
    clamped_keys = [max(min_v, min(max_v, k)) for k, min_v, max_v in zip(key_parts, min_vals, max_vals)]
    key = ",".join(map(str, clamped_keys))
    return table.get(key, 1) # Default to 1 if key somehow still invalid

def getScoreA(trunk: int, neck: int, leg: int, loadKgInput: float) -> int:
    # ↓↓↓ Indentation required ↓↓↓
    base_score = lookup_score(tableA_Lookup, [trunk, neck, leg], [1, 1, 1], [6, 4, 4]) # Max vals from calc funcs
    return base_score + calc_load_score(loadKgInput)

def getScoreB(upperArm: int, forearm: int, wrist: int, coupling: int) -> int:
     # ↓↓↓ Indentation required ↓↓↓
    base_score = lookup_score(tableB_Lookup, [upperArm, forearm, wrist], [1, 1, 1], [6, 2, 3]) # Max vals from calc funcs
    return base_score + coupling # coupling score is 0, 1, 2, or 3

def getTableCScore(scoreA: int, scoreB: int) -> int:
    # ↓↓↓ Indentation required ↓↓↓
    # REBA Table C max values are typically 12
    return lookup_score(tableC_Lookup, [scoreA, scoreB], [1, 1], [12, 12])

def get_risk_level(score: int) -> str:
    # ↓↓↓ Indentation required ↓↓↓
    if score == 1: return "無視できる (Negligible)"
    elif 2 <= score <= 3: return "低リスク (Low)"
    elif 4 <= score <= 7: return "中リスク (Medium)"
    elif 8 <= score <= 10: return "高リスク (High)"
    else: return "非常に高リスク (Very High)" # Score 11-15

# -----------------------------
# 最終REBAスコア算出関数
# -----------------------------
def get_final_reba_score(landmarks: List[Landmark], calib: CalibrationInputs) -> Dict[str, Any]:
    # ↓↓↓ Indentation required ↓↓↓
    try:
        angles = compute_all_angles(landmarks)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error computing angles: {e}")
        raise HTTPException(status_code=500, detail="Error computing body angles.")

    # Convert flags from input floats/ints
    neck_twist_flag = calib.neckRotation > 0
    neck_bend_flag = calib.neckLateralBending > 0
    # Use calculated trunk rotation angle with a threshold (e.g., 5 degrees)
    trunk_twist_flag = abs(angles.get("trunkRotationAngle", 0)) >= 5
    trunk_bend_flag = calib.trunkLateralFlexion > 0

    neckScore = calc_neck_score(angles.get("neckAngle", 90), neck_twist_flag, neck_bend_flag) # Default angle if missing
    trunkScore = calc_trunk_score(angles.get("trunkFlexionAngle", 0), trunk_twist_flag, trunk_bend_flag)

    # Side-specific evaluation
    if calib.filmingSide == "left":
        legScore = calc_leg_score_unified(calib.postureCategory, angles.get("leftKneeAngle", 180))
        upperArmAngle_eval = angles.get("leftUpperArmAngle", 90)
        forearmAngle_eval = angles.get("leftElbowAngle", 90)
        wristAngle_eval = angles.get("leftWristAngle", 0) # Still approximate
    else: # right
        legScore = calc_leg_score_unified(calib.postureCategory, angles.get("rightKneeAngle", 180))
        upperArmAngle_eval = angles.get("rightUpperArmAngle", 90)
        forearmAngle_eval = angles.get("rightElbowAngle", 90)
        wristAngle_eval = angles.get("rightWristAngle", 0) # Still approximate

    # Score A
    scoreA = getScoreA(trunkScore, neckScore, legScore, calib.loadForce)

    # Score B
    upperArmScore = calc_upper_arm_score(upperArmAngle_eval, calib.upperArmCorrection, calib.shoulderElevation, calib.gravityAssist)
    forearmScore = calc_forearm_score(forearmAngle_eval)
    wristScore = calc_wrist_score(wristAngle_eval, calib.wristCorrection)
    scoreB = getScoreB(upperArmScore, forearmScore, wristScore, calib.coupling)

    # Final Score
    tableCScore = getTableCScore(scoreA, scoreB)
    activityScore = calib.staticPosture + calib.repetitiveMovement + calib.unstableMovement
    finalScore = tableCScore + activityScore
    finalScore = max(1, min(15, finalScore)) # Clamp to 1-15
    riskLevel = get_risk_level(finalScore)

    response_data = {
        "final_score": finalScore,
        "risk_level": riskLevel,
        "computed_angles": angles,
        "intermediate_scores": {
             "neck": neckScore, "trunk": trunkScore, "leg": legScore,
             "upperArm": upperArmScore, "forearm": forearmScore, "wrist": wristScore,
             "scoreA": scoreA, "scoreB": scoreB, "tableC": tableCScore, "activity": activityScore
        }
    }
    print("DEBUG: Returning data:", response_data) # Debug log
    return response_data

# -----------------------------
# API エンドポイント
# -----------------------------
@app.post("/compute_reba")
async def compute_reba_endpoint(input_data: REBAInput):
    # ↓↓↓ Indentation required ↓↓↓
    # Pydantic validation happens automatically based on type hint
    try:
        result = get_final_reba_score(input_data.landmarks, input_data.calibInputs)
        return result
    except HTTPException as e:
         raise e # Re-raise known HTTP errors
    except ValidationError as e: # Should be caught by FastAPI, but handle just in case
         raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
         print(f"ERROR: Unexpected error during REBA calculation: {e}")
         import traceback
         traceback.print_exc()
         raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# -----------------------------
# CORS ミドルウェア設定
# -----------------------------
# Ensure this matches your frontend deployment URL
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "https://reba-1.onrender.com", # Your frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"], # Allow necessary methods
    allow_headers=["*"], # Allow all headers
)

# -----------------------------
# ルートエンドポイント
# -----------------------------
@app.get("/")
async def read_root():
    # ↓↓↓ Indentation required ↓↓↓
    return {"message": "REBA Evaluation API is running"}

