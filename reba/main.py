from fastapi import FastAPI, HTTPException
# Depends は Pydantic v2 + FastAPI 最新版では不要になる場合あり
# from fastapi import Depends
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Dict, Any
import math
# CORS設定用
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="REBA Evaluation API")

# -----------------------------
# モデル定義 (Pydantic)
# -----------------------------
class Landmark(BaseModel):
    x: float
    y: float
    z: float = 0.0
    visibility: float = Field(default=1.0, ge=0.0, le=1.0) # visibilityも受け取るように変更(ge=以上, le=以下)

class CalibrationInputs(BaseModel):
    filmingSide: str = Field(..., description="left or right")
    neckRotation: float
    neckLateralBending: float
    trunkLateralFlexion: float
    loadForce: float
    postureCategory: str
    upperArmCorrection: float
    shoulderElevation: float
    gravityAssist: float
    wristCorrection: float
    staticPosture: int
    repetitiveMovement: int
    unstableMovement: int
    coupling: int # 値は 0, 1, 2, 3 を想定

    # (9) バリデーション追加
    @validator('coupling')
    def coupling_must_be_valid(cls, v):
        if not 0 <= v <= 3: # HTML側で 0-3 を送るようにしたので範囲を修正
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

    # 必要に応じて他の数値範囲バリデータを追加
    # @validator('loadForce')
    # def load_non_negative(cls, v):
    #     if v < 0: raise ValueError('Load cannot be negative')
    #     return v

class REBAInput(BaseModel):
    landmarks: List[Landmark]
    calibInputs: CalibrationInputs

# -----------------------------
# 角度計算ユーティリティ
# -----------------------------
def calculate_angle(p1: Dict[str, float], p2: Dict[str, float], p3: Dict[str, float], min_visibility: float = 0.5) -> float:
    """ Calculates the angle between three points (p1, p2, p3), where p2 is the vertex. """
    # Visibility check (optional but recommended)
    if p1.get('visibility', 1.0) < min_visibility or \
       p2.get('visibility', 1.0) < min_visibility or \
       p3.get('visibility', 1.0) < min_visibility:
        return 0.0 # Return 0 or perhaps None/NaN if visibility is too low

    # Vector subtraction
    v1 = {"x": p1["x"] - p2["x"], "y": p1["y"] - p2["y"]}
    v2 = {"x": p3["x"] - p2["x"], "y": p3["y"] - p2["y"]}

    # Dot product
    dot = v1["x"] * v2["x"] + v1["y"] * v2["y"]

    # Vector magnitudes
    mag1 = math.sqrt(v1["x"]**2 + v1["y"]**2)
    mag2 = math.sqrt(v2["x"]**2 + v2["y"]**2)

    # Prevent division by zero
    if mag1 * mag2 == 0:
        return 0.0

    # Calculate cosine of angle, clamped to [-1, 1] for numerical stability
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))

    # Calculate angle in degrees
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)

def angle_with_vertical(p1: Dict[str, float], p2: Dict[str, float], min_visibility: float = 0.5) -> float:
    """ Calculates the angle of the vector from p1 to p2 with the downward vertical axis. """
    if p1.get('visibility', 1.0) < min_visibility or p2.get('visibility', 1.0) < min_visibility:
        return 0.0 # Or handle differently

    vector = {"x": p2["x"] - p1["x"], "y": p2["y"] - p1["y"]} # Vector p1 -> p2
    vertical = {"x": 0, "y": 1} # Downward vertical vector (assuming Y increases downwards in normalized coords)

    dot = vector["x"] * vertical["x"] + vector["y"] * vertical["y"]
    norm = math.sqrt(vector["x"]**2 + vector["y"]**2)

    if norm == 0:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / norm))
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad) # Angle with vertical (0=vertical, 90=horizontal)

# -----------------------------
# ランドマークから角度計算
# -----------------------------
def compute_all_angles(landmarks: List[Landmark]) -> Dict[str, float]:
    # ランドマークインデックス (MediaPipe Pose)
    lm_indices = {
        "Nose": 0, "L_Shoulder": 11, "R_Shoulder": 12, "L_Elbow": 13, "R_Elbow": 14,
        "L_Wrist": 15, "R_Wrist": 16, "L_Hip": 23, "R_Hip": 24, "L_Knee": 25,
        "R_Knee": 26, "L_Ankle": 27, "R_Ankle": 28,
        # 手首の角度計算にはより詳細なランドマークが必要だが、仮で手首を使用
        # "L_Index": 19, "R_Index": 20 # Note: These are finger tips
    }
    max_index = max(lm_indices.values())
    if len(landmarks) <= max_index:
         raise HTTPException(status_code=400, detail=f"Not enough landmarks provided. Need at least {max_index + 1}")

    # Pydanticモデルを辞書リストに変換 (v1: .dict(), v2: .model_dump())
    lm = [landmark.model_dump() if hasattr(landmark, 'model_dump') else landmark.dict() for landmark in landmarks]

    # --- Calculate Midpoints ---
    shoulder_mid = {"x": (lm[lm_indices["L_Shoulder"]]["x"] + lm[lm_indices["R_Shoulder"]]["x"]) / 2,
                    "y": (lm[lm_indices["L_Shoulder"]]["y"] + lm[lm_indices["R_Shoulder"]]["y"]) / 2,
                    "visibility": min(lm[lm_indices["L_Shoulder"]].get('visibility',1.0), lm[lm_indices["R_Shoulder"]].get('visibility',1.0))}
    hip_mid = {"x": (lm[lm_indices["L_Hip"]]["x"] + lm[lm_indices["R_Hip"]]["x"]) / 2,
               "y": (lm[lm_indices["L_Hip"]]["y"] + lm[lm_indices["R_Hip"]]["y"]) / 2,
               "visibility": min(lm[lm_indices["L_Hip"]].get('visibility',1.0), lm[lm_indices["R_Hip"]].get('visibility',1.0))}

    # --- Neck Angle (Flexion/Extension) ---
    # Using angle of vector from shoulder midpoint to nose relative to vertical
    neckAngle = angle_with_vertical(shoulder_mid, lm[lm_indices["Nose"]])

    # --- Trunk Angle (Flexion/Extension) ---
    # Using angle of vector from hip midpoint to shoulder midpoint relative to vertical
    trunkFlexionAngle = angle_with_vertical(hip_mid, shoulder_mid)

    # --- Trunk Rotation (Highly Simplified 2D Estimate) ---
    # Angle difference between shoulder line and hip line relative to horizontal
    # This is very approximate for actual trunk rotation.
    shoulder_dx = lm[lm_indices["R_Shoulder"]]["x"] - lm[lm_indices["L_Shoulder"]]["x"]
    shoulder_dy = lm[lm_indices["R_Shoulder"]]["y"] - lm[lm_indices["L_Shoulder"]]["y"]
    hip_dx = lm[lm_indices["R_Hip"]]["x"] - lm[lm_indices["L_Hip"]]["x"]
    hip_dy = lm[lm_indices["R_Hip"]]["y"] - lm[lm_indices["L_Hip"]]["y"]
    shoulder_angle_rad = math.atan2(shoulder_dy, shoulder_dx)
    hip_angle_rad = math.atan2(hip_dy, hip_dx)
    trunkRotationAngle = math.degrees(shoulder_angle_rad - hip_angle_rad)
    trunkRotationAngle = (trunkRotationAngle + 180) % 360 - 180 # Normalize to -180 to 180

    # --- Limb Angles (Internal Angles) ---
    # (8) Calculate angles for both sides
    # Upper Arm (Angle at shoulder relative to torso is complex in 2D)
    # REBA needs angle relative to trunk. Calculating Hip-Shoulder-Elbow angle is an alternative but not exact REBA def.
    # We stick to Hip-Shoulder-Elbow for now.
    leftUpperArmAngle = calculate_angle(lm[lm_indices["L_Hip"]], lm[lm_indices["L_Shoulder"]], lm[lm_indices["L_Elbow"]])
    rightUpperArmAngle = calculate_angle(lm[lm_indices["R_Hip"]], lm[lm_indices["R_Shoulder"]], lm[lm_indices["R_Elbow"]])
    # Elbow (Internal angle)
    leftElbowAngle = calculate_angle(lm[lm_indices["L_Shoulder"]], lm[lm_indices["L_Elbow"]], lm[lm_indices["L_Wrist"]])
    rightElbowAngle = calculate_angle(lm[lm_indices["R_Shoulder"]], lm[lm_indices["R_Elbow"]], lm[lm_indices["R_Wrist"]])
    # Wrist (Internal angle - Very approximate for REBA Flexion/Extension)
    # Using Elbow-Wrist-Wrist might be 0, need a third point. Using midpoint of fingers? Or just rely on user input?
    # Let's return 0 for wrist angle from calculation for now, suggest user input.
    leftWristAngle = 0.0 # calculate_angle(lm[lm_indices["L_Elbow"]], lm[lm_indices["L_Wrist"]], lm[lm_indices["L_Index"]]) # Example, likely inaccurate
    rightWristAngle = 0.0 # calculate_angle(lm[lm_indices["R_Elbow"]], lm[lm_indices["R_Wrist"]], lm[lm_indices["R_Index"]])
    # Knee (Internal angle)
    leftKneeAngle = calculate_angle(lm[lm_indices["L_Hip"]], lm[lm_indices["L_Knee"]], lm[lm_indices["L_Ankle"]])
    rightKneeAngle = calculate_angle(lm[lm_indices["R_Hip"]], lm[lm_indices["R_Knee"]], lm[lm_indices["R_Ankle"]])

    return {
        "neckAngle": neckAngle,
        "trunkFlexionAngle": trunkFlexionAngle,
        "trunkRotationAngle": trunkRotationAngle,
        "leftUpperArmAngle": leftUpperArmAngle,
        "rightUpperArmAngle": rightUpperArmAngle,
        "leftElbowAngle": leftElbowAngle,
        "rightElbowAngle": rightElbowAngle,
        "leftWristAngle": leftWristAngle, # Mark as approximate/unreliable
        "rightWristAngle": rightWristAngle, # Mark as approximate/unreliable
        "leftKneeAngle": leftKneeAngle,
        "rightKneeAngle": rightKneeAngle
    }

# -----------------------------
# REBAスコア計算関数群
# (Note: Some score interpretations below might need verification against official REBA sheet)
# -----------------------------
def calc_neck_score(neckAngle: float, rotationFlag: bool, sideBendFlag: bool) -> int:
    # Assuming neckAngle 0 is upright, positive is flexion
    if neckAngle <= 20: # 0-20 degrees flexion
        base = 1
    else: # > 20 degrees flexion or in extension
        base = 2 # Need to handle extension case if angle definition allows negative for extension
    # REBA: Add +1 for Twist, Add +1 for Side Bend
    add = 0
    if rotationFlag: add += 1
    if sideBendFlag: add += 1
    return base + add # Max score 4

def calc_trunk_score(trunkFlexAngle: float, rotationFlag: bool, sideBendFlag: bool) -> int:
    # Assuming trunkFlexAngle 0 is upright, positive is flexion
    if trunkFlexAngle <= 5: base = 1 # 0 degrees (Upright)
    elif trunkFlexAngle <= 20: base = 2 # 1-20 degrees flexion
    elif trunkFlexAngle <= 60: base = 3 # 21-60 degrees flexion
    else: base = 4 # > 60 degrees flexion
    # Need to handle trunk extension case based on angle definition
    # REBA: Add +1 for Twist, Add +1 for Side Bend
    add = 0
    if rotationFlag: add += 1
    if sideBendFlag: add += 1
    return base + add # Max score 6

def calc_leg_score_unified(postureCategory: str, kneeFlexAngle: float) -> int:
    if postureCategory == "sittingWalking": # REBA: Sitting (Assume legs supported)
        return 1
    # REBA: Standing
    base = 1 if postureCategory == "standingBoth" else 2 # Bilateral weight bearing vs Unilateral/Unstable
    # Add for Knee flexion (REBA: 30-60 deg = +1, >60 deg = +2)
    # Assuming kneeFlexAngle is internal angle (0=straight)
    flex = 180 - kneeFlexAngle # Estimate flexion from internal angle
    add = 0
    if 30 <= flex <= 60:
        add = 1
    elif flex > 60:
        add = 2
    return base + add # Max score 4

def calc_upper_arm_score(upperArmAngle: float, upperArmCorrection: float, shoulderElevation: float, gravityAssist: float) -> int:
    # REBA: Angle relative to trunk (0=alongside, positive=flexion/abduction, negative=extension)
    # Calculated angle (Hip-Shoulder-Elbow) is not directly this. Using it as a proxy.
    # Needs refinement for accurate REBA. Assuming angle relates somewhat to flexion.
    flex_angle = upperArmAngle - 90 # Very rough approximation to get 0 = vertical-ish
    if flex_angle <= 20: base = 1 # 0-20 deg flexion
    elif flex_angle <= 45: base = 2 # 21-45 deg
    elif flex_angle <= 90: base = 3 # 46-90 deg
    else: base = 4 # > 90 deg
    # Extension case needs handling based on angle definition
    # REBA Corrections: Add +1 if abducted or rotated, Add +1 if shoulder raised, Subtract -1 if arm supported
    # Mapping: upperArmCorrection=abducted/rotated?, shoulderElevation=raised?, gravityAssist=supported?
    corrected = base + upperArmCorrection + shoulderElevation + gravityAssist
    return max(1, min(6, int(round(corrected)))) # Max score 6

def calc_forearm_score(elbowAngle: float) -> int:
    # REBA: Angle relative to upper arm. 60-100 deg = 1, else 2.
    # Assuming elbowAngle is internal angle (Shoulder-Elbow-Wrist).
    if 60 <= elbowAngle <= 100:
        return 1
    else:
        return 2 # Max score 2

def calc_wrist_score(wristAngle: float, wristCorrection: float) -> int:
    # REBA: Flexion/Extension angle. 0-15 deg = 1, >15 deg = 2.
    # Calculated angle is highly approximate, if calculated at all. Defaulting to base=1.
    base = 1 # Assume neutral or rely on user input?
    # If wristAngle could approximate flexion: base = 1 if wristAngle <= 15 else 2
    # REBA Correction: Add +1 for twist or deviation
    score = base + wristCorrection
    return max(1, min(3, int(round(score)))) # Max score 3

def calc_load_score(loadKg: float) -> int:
    # REBA: <5kg = +0, 5-10kg = +1, >10kg = +2. Shock/Rapid build up = +1
    # Assuming loadKg maps directly and no shock force for now.
    if loadKg < 5: return 0
    elif loadKg <= 10: return 1
    else: return 2 # Max score 2

# ルックアップテーブル (変更なしだが、入力スコア範囲に注意)
tableA_Lookup = { "1,1,1":1, "1,1,2":2, "1,1,3":3, "1,1,4":4, "1,2,1":1, "1,2,2":2, "1,2,3":3, "1,2,4":4, "1,3,1":3, "1,3,2":3, "1,3,3":5, "1,3,4":6, "2,1,1":2, "2,1,2":3, "2,1,3":4, "2,1,4":5, "2,2,1":3, "2,2,2":4, "2,2,3":5, "2,2,4":6, "2,3,1":4, "2,3,2":5, "2,3,3":6, "2,3,4":7, "3,1,1":2, "3,1,2":4, "3,1,3":5, "3,1,4":6, "3,2,1":4, "3,2,2":5, "3,2,3":6, "3,2,4":7, "3,3,1":5, "3,3,2":6, "3,3,3":7, "3,3,4":8, "4,1,1":3, "4,1,2":5, "4,1,3":6, "4,1,4":7, "4,2,1":5, "4,2,2":6, "4,2,3":7, "4,2,4":8, "4,3,1":6, "4,3,2":7, "4,3,3":8, "4,3,4":9, "5,1,1":4, "5,1,2":6, "5,1,3":7, "5,1,4":8, "5,2,1":6, "5,2,2":7, "5,2,3":8, "5,2,4":9, "5,3,1":7, "5,3,2":8, "5,3,3":9, "5,3,4":9, "6,1,1":5,"6,1,2":7,"6,1,3":8,"6,1,4":9,"6,2,1":7,"6,2,2":8,"6,2,3":9,"6,2,4":9,"6,3,1":8,"6,3,2":9,"6,3,3":9,"6,3,4":9 } # Trunk max 6
tableB_Lookup = { "1,1,1":1, "1,1,2":2, "1,1,3":2, "1,2,1":1, "1,2,2":2, "1,2,3":3, "2,1,1":1, "2,1,2":2, "2,1,3":3, "2,2,1":2, "2,2,2":3, "2,2,3":4, "3,1,1":3, "3,1,2":4, "3,1,3":5, "3,2,1":4, "3,2,2":5, "3,2,3":5, "4,1,1":4, "4,1,2":5, "4,1,3":5, "4,2,1":5, "4,2,2":6, "4,2,3":7, "5,1,1":6, "5,1,2":7, "5,1,3":8, "5,2,1":7, "5,2,2":8, "5,2,3":8, "6,1,1":7, "6,1,2":8, "6,1,3":8, "6,2,1":8, "6,2,2":9, "6,2,3":9 } # Wrist max 3
tableC_Lookup = { "1,1":1,"1,2":1,"1,3":1,"1,4":2,"1,5":3,"1,6":3,"1,7":4,"1,8":5,"1,9":6,"1,10":7,"1,11":7,"1,12":7, "2,1":1,"2,2":2,"2,3":2,"2,4":3,"2,5":4,"2,6":4,"2,7":5,"2,8":6,"2,9":6,"2,10":7,"2,11":7,"2,12":8, "3,1":2,"3,2":3,"3,3":3,"3,4":4,"3,5":5,"3,6":6,"3,7":7,"3,8":7,"3,9":7,"3,10":8,"3,11":8,"3,12":8, "4,1":3,"4,2":4,"4,3":4,"4,4":4,"4,5":5,"4,6":6,"4,7":7,"4,8":8,"4,9":8,"4,10":9,"4,11":9,"4,12":9, "5,1":4,"5,2":4,"5,3":4,"5,4":5,"5,5":6,"5,6":7,"5,7":8,"5,8":8,"5,9":9,"5,10":9,"5,11":9,"5,12":9, "6,1":6,"6,2":6,"6,3":6,"6,4":7,"6,5":8,"6,6":8,"6,7":9,"6,8":9,"6,9":10,"6,10":10,"6,11":10,"6,12":10, "7,1":7,"7,2":7,"7,3":7,"7,4":8,"7,5":9,"7,6":9,"7,7":9,"7,8":10,"7,9":10,"7,10":11,"7,11":11,"7,12":11, "8,1":8,"8,2":8,"8,3":8,"8,4":9,"8,5":10,"8,6":10,"8,7":10,"8,8":10,"8,9":11,"8,10":11,"8,11":11,"8,12":12, "9,1":9,"9,2":9,"9,3":9,"9,4":10,"9,5":10,"9,6":11,"9,7":11,"9,8":11,"9,9":12,"9,10":12,"9,11":12,"9,12":12, "10,1":10,"10,2":10,"10,3":10,"10,4":11,"10,5":11,"10,6":11,"10,7":12,"10,8":12,"10,9":12,"10,10":12,"10,11":12,"10,12":12, "11,1":11,"11,2":11,"11,3":11,"11,4":12,"11,5":12,"11,6":12,"11,7":12,"11,8":12,"11,9":12,"11,10":12,"11,11":12,"11,12":12, "12,1":12,"12,2":12,"12,3":12,"12,4":12,"12,5":12,"12,6":12,"12,7":12,"12,8":12,"12,9":12,"12,10":12,"12,11":12,"12,12":12 }

# Helper to look up in table, handling potential out-of-range scores
def lookup_score(table: Dict[str, int], key_parts: List[int], min_vals: List[int], max_vals: List[int]) -> int:
    # Clamp keys to valid range defined by min/max_vals
    clamped_keys = [max(min_v, min(max_v, k)) for k, min_v, max_v in zip(key_parts, min_vals, max_vals)]
    key = ",".join(map(str, clamped_keys))
    return table.get(key, 1) # Default to 1 if key somehow still invalid

def getScoreA(trunk: int, neck: int, leg: int, loadKg: float) -> int:
    # Define valid ranges for Table A keys based on score functions
    base_score = lookup_score(tableA_Lookup, [trunk, neck, leg], [1, 1, 1], [6, 4, 4])
    return base_score + calc_load_score(loadKg)

def getScoreB(upperArm: int, forearm: int, wrist: int, coupling: int) -> int:
     # Define valid ranges for Table B keys
    base_score = lookup_score(tableB_Lookup, [upperArm, forearm, wrist], [1, 1, 1], [6, 2, 3])
    # Coupling score is added AFTER table lookup
    return base_score + coupling # coupling is 0, 1, 2, or 3

def getTableCScore(scoreA: int, scoreB: int) -> int:
    # Define valid ranges for Table C keys (REBA sheet usually goes up to 12)
    return lookup_score(tableC_Lookup, [scoreA, scoreB], [1, 1], [12, 12])

def get_risk_level(score: int) -> str:
    # Standard REBA risk levels
    if score == 1: return "無視できる (Negligible)"
    elif 2 <= score <= 3: return "低リスク (Low)"
    elif 4 <= score <= 7: return "中リスク (Medium)"
    elif 8 <= score <= 10: return "高リスク (High)"
    else: return "非常に高リスク (Very High)" # Score 11-15

# -----------------------------
# 最終REBAスコア算出関数
# -----------------------------
def get_final_reba_score(landmarks: List[Landmark], calib: CalibrationInputs) -> Dict[str, Any]:
    try:
        angles = compute_all_angles(landmarks)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error computing angles: {e}")
        raise HTTPException(status_code=500, detail="Error computing body angles.")

    # Neck and Trunk Scores (Use flags from inputs)
    # Note: REBA twist/side bend flags might need different interpretation than just > 0
    neck_twist_flag = calib.neckRotation > 0
    neck_bend_flag = calib.neckLateralBending > 0
    trunk_twist_flag = abs(angles["trunkRotationAngle"]) >= 5 # Example threshold for twist flag
    trunk_bend_flag = calib.trunkLateralFlexion > 0

    neckScore = calc_neck_score(angles["neckAngle"], neck_twist_flag, neck_bend_flag)
    trunkScore = calc_trunk_score(angles["trunkFlexionAngle"], trunk_twist_flag, trunk_bend_flag)

    # (8) Side-specific evaluation
    if calib.filmingSide == "left":
        legScore = calc_leg_score_unified(calib.postureCategory, angles["leftKneeAngle"])
        upperArmAngle_eval = angles["leftUpperArmAngle"]
        forearmAngle_eval = angles["leftElbowAngle"]
        wristAngle_eval = angles["leftWristAngle"] # Still approximate
    else: # right
        legScore = calc_leg_score_unified(calib.postureCategory, angles["rightKneeAngle"])
        upperArmAngle_eval = angles["rightUpperArmAngle"]
        forearmAngle_eval = angles["rightElbowAngle"]
        wristAngle_eval = angles["rightWristAngle"] # Still approximate

    # Score A (Check max possible score based on individual components)
    scoreA = getScoreA(trunkScore, neckScore, legScore, calib.loadForce)

    # Score B
    # Upper Arm score needs careful check of angle definition vs REBA standard
    upperArmScore = calc_upper_arm_score(upperArmAngle_eval,
                                         calib.upperArmCorrection,
                                         calib.shoulderElevation,
                                         calib.gravityAssist)
    forearmScore = calc_forearm_score(forearmAngle_eval)
    # Wrist score depends heavily on input or better angle calc
    wristScore = calc_wrist_score(wristAngle_eval, calib.wristCorrection) # wristAngle_eval is often 0 here
    scoreB = getScoreB(upperArmScore, forearmScore, wristScore, calib.coupling) # coupling is added here

    # Final Score
    tableCScore = getTableCScore(scoreA, scoreB)
    # Activity Score: Sum of 3 flags (+1 each if applicable)
    activityScore = calib.staticPosture + calib.repetitiveMovement + calib.unstableMovement
    finalScore = tableCScore + activityScore

    finalScore = max(1, min(15, finalScore)) # REBA score capped at 15
    riskLevel = get_risk_level(finalScore)

    return {
        "final_score": finalScore,
        "risk_level": riskLevel,
        "computed_angles": angles,
        "intermediate_scores": {
             "neck": neckScore, "trunk": trunkScore, "leg": legScore,
             "upperArm": upperArmScore, "forearm": forearmScore, "wrist": wristScore,
             "scoreA": scoreA, "scoreB": scoreB, "tableC": tableCScore, "activity": activityScore
        }
    }

# -----------------------------
# API エンドポイント
# -----------------------------
@app.post("/compute_reba")
async def compute_reba_endpoint(input_data: REBAInput):
    # Pydantic validation runs automatically based on type hint
    try:
        result = get_final_reba_score(input_data.landmarks, input_data.calibInputs)
        return result
    except HTTPException as e:
         # Re-raise exceptions with status codes (e.g., from angle calc)
         raise e
    except ValidationError as e: # Catch Pydantic validation errors explicitly if needed
         # FastAPI usually handles this automatically with 422
         raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
         # Catch unexpected errors during calculation
         print(f"ERROR: Unexpected error during REBA calculation: {e}")
         import traceback
         traceback.print_exc() # Print stack trace for debugging
         raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# -----------------------------
# CORS ミドルウェア設定 (重要！)
# -----------------------------
# !! デプロイ後にフロントエンドのURLに合わせて origins を設定してください !!
origins = [
    "http://localhost", # ローカルテスト用 (残しておいても良い)
    "http://127.0.0.1", # ローカルテスト用 (残しておいても良い)
    # ↓↓↓ ここに Render でデプロイした「静的サイト」のURLを追加・変更します ↓↓↓
    "https://reba-1.onrender.com", # 例: スクリーンショットに表示されていたURL (これがフロントエンド用か確認してください)
    # "https://your-reba-frontend-static-site-name.onrender.com", # Render Static Site URL の正しい形式の例
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # ★ このリストにフロントエンドのURLが含まれている必要があります
    allow_credentials=True,
    allow_methods=["POST", "GET"], # APIで使うHTTPメソッドを許可
    allow_headers=["*"], # すべてのリクエストヘッダーを許可
)


# -----------------------------
# (オプション) ルートエンドポイント
# -----------------------------
@app.get("/")
async def read_root():
    return {"message": "REBA Evaluation API is running"}

