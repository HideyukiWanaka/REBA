# main.py (最終版 - 2D計算、手首スコアユーザー入力反映)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Dict, Any, Optional
import math
from fastapi.middleware.cors import CORSMiddleware # CORS用

app = FastAPI(title="REBA Evaluation API")

# -----------------------------
# モデル定義 (Pydantic)
# -----------------------------
class Landmark(BaseModel):
    x: float
    y: float
    z: Optional[float] = 0.0
    visibility: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class CalibrationInputs(BaseModel):
    filmingSide: str = Field(..., description="left or right")
    neckRotation: float # Flag: >0 means rotated
    neckLateralBending: float # Flag: >0 means bent
    trunkLateralFlexion: float # Flag: >0 means bent
    loadForce: float # Represents score bracket start (0, 5, 11)
    postureCategory: str # "standingBoth", "standingOne", "sittingWalking"
    upperArmCorrection: float # Abduction/Rotation flag (0 or 1)
    shoulderElevation: float # Shoulder raised flag (0 or 1)
    gravityAssist: float # Arm supported flag (0 or -1)
    wristCorrection: float # Deviation/Twist flag (0 or 1)
    wristBaseScore: int    # ★ Base score from user input (1 or 2) ★
    staticPosture: int # Activity score flag (0 or 1)
    repetitiveMovement: int # Activity score flag (0 or 1)
    unstableMovement: int # Activity score flag (0 or 1)
    coupling: int # Coupling score addition (0, 1, 2, or 3)

    @validator('coupling')
    def coupling_must_be_valid(cls, v):
        if not 0 <= v <= 3: raise ValueError('Coupling score must be between 0 and 3')
        return v

    @validator('filmingSide')
    def side_must_be_valid(cls, v):
        if v not in ["left", "right"]: raise ValueError('Filming side must be "left" or "right"')
        return v

    @validator('postureCategory')
    def posture_must_be_valid(cls, v):
        if v not in ["standingBoth", "standingOne", "sittingWalking"]: raise ValueError('Invalid posture category')
        return v

    @validator('wristBaseScore')
    def wrist_base_score_valid(cls, v):
        if v not in [1, 2]: raise ValueError('Wrist Base Score must be 1 or 2')
        return v

class REBAInput(BaseModel):
    landmarks: List[Landmark]
    calibInputs: CalibrationInputs

# -----------------------------
# 2D Vector Calculation Helpers
# -----------------------------
def parse_point(p: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    return p if p and isinstance(p, dict) else None

def vec_subtract_2d(p1_data: Dict, p2_data: Dict) -> Optional[Dict[str, float]]:
    p1 = parse_point(p1_data); p2 = parse_point(p2_data)
    if p1 and p2 and 'x' in p1 and 'y' in p1 and 'x' in p2 and 'y' in p2:
        return {"x": p1['x'] - p2['x'], "y": p1['y'] - p2['y']}
    return None

def vec_dot_2d(v1: Dict, v2: Dict) -> float:
    return v1.get('x', 0.0) * v2.get('x', 0.0) + v1.get('y', 0.0) * v2.get('y', 0.0)

def vec_cross_2d(v1: Dict, v2: Dict) -> float:
    return v1.get('x', 0.0) * v2.get('y', 0.0) - v1.get('y', 0.0) * v2.get('x', 0.0)

def vec_magnitude_2d(v: Optional[Dict[str, float]]) -> float:
    if not v: return 0.0
    mag_sq = v.get('x', 0.0)**2 + v.get('y', 0.0)**2
    return math.sqrt(mag_sq) if mag_sq > 1e-9 else 0.0

def angle_between_2d_vectors(v1_data: Dict, v2_data: Dict) -> float:
    v1 = parse_point(v1_data); v2 = parse_point(v2_data)
    if not v1 or not v2: return 0.0
    mag1 = vec_magnitude_2d(v1); mag2 = vec_magnitude_2d(v2)
    if mag1 * mag2 < 1e-9: return 0.0
    dot = vec_dot_2d(v1, v2)
    cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    if math.isnan(cos_theta): cos_theta = 1.0 if dot > 0 else -1.0
    try: return math.degrees(math.acos(cos_theta))
    except ValueError: return 0.0

def angle_with_vertical(p1_data: Dict, p2_data: Dict, min_visibility: float = 0.5) -> float:
    p1 = parse_point(p1_data); p2 = parse_point(p2_data)
    if not p1 or not p2 or p1.get('visibility', 1.0) < min_visibility or p2.get('visibility', 1.0) < min_visibility: return 0.0
    vector = vec_subtract_2d(p2, p1)
    if not vector: return 0.0
    vertical = {"x": 0, "y": 1} # Y increases downwards
    return angle_between_2d_vectors(vector, vertical)

def calculate_angle(p1_data: Dict, p2_data: Dict, p3_data: Dict, min_visibility: float = 0.5) -> float:
    p1 = parse_point(p1_data); p2 = parse_point(p2_data); p3 = parse_point(p3_data)
    if not p1 or not p2 or not p3 or \
       p1.get('visibility', 1.0) < min_visibility or \
       p2.get('visibility', 1.0) < min_visibility or \
       p3.get('visibility', 1.0) < min_visibility: return 0.0
    v1 = vec_subtract_2d(p1, p2); v2 = vec_subtract_2d(p3, p2)
    if not v1 or not v2: return 0.0
    return angle_between_2d_vectors(v1, v2)

# -----------------------------
# 角度計算 (Pure 2D)
# -----------------------------
def compute_all_angles(landmarks: List[Landmark], filming_side: str) -> Dict[str, Any]:
    lm_indices = { "Nose": 0, "L_Shoulder": 11, "R_Shoulder": 12, "L_Elbow": 13, "R_Elbow": 14, "L_Wrist": 15, "R_Wrist": 16, "L_Hip": 23, "R_Hip": 24, "L_Knee": 25, "R_Knee": 26, "L_Ankle": 27, "R_Ankle": 28 }
    max_index = max(lm_indices.values())
    if len(landmarks) <= max_index: raise HTTPException(status_code=400, detail=f"Not enough landmarks ({len(landmarks)}). Need {max_index + 1}")

    lm = [lm.model_dump() if hasattr(lm, 'model_dump') else lm.dict() for lm in landmarks]
    min_vis = 0.5

    # Midpoints
    ls = lm[lm_indices["L_Shoulder"]]; rs = lm[lm_indices["R_Shoulder"]]
    lh = lm[lm_indices["L_Hip"]]; rh = lm[lm_indices["R_Hip"]]
    shoulder_mid = {"x": (ls['x'] + rs['x']) / 2, "y": (ls['y'] + rs['y']) / 2, "visibility": min(ls.get('visibility',0), rs.get('visibility',0))} if ls and rs else None
    hip_mid = {"x": (lh['x'] + rh['x']) / 2, "y": (lh['y'] + rh['y']) / 2, "visibility": min(lh.get('visibility',0), rh.get('visibility',0))} if lh and rh else None
    if not shoulder_mid or not hip_mid or shoulder_mid.get('visibility',0) < min_vis or hip_mid.get('visibility',0) < min_vis:
         raise HTTPException(status_code=400, detail="Cannot compute midpoints (shoulder/hip).")
    nose = lm[lm_indices["Nose"]]

    # Helper: Extension Flag from Vertical + X-Component
    def get_extension_flag_from_vertical(p1: Dict, p2: Dict, side: str, angle_mag_vert: float) -> bool:
        is_ext = False
        p1_p = parse_point(p1); p2_p = parse_point(p2)
        if not p1_p or not p2_p or p1_p.get('visibility', 0) < min_vis or p2_p.get('visibility', 0) < min_vis: return False
        vector_x = p2_p.get('x', 0.5) - p1_p.get('x', 0.5)
        x_threshold = 0.02
        if angle_mag_vert > 5:
            if side == "left": is_ext = vector_x > x_threshold
            elif side == "right": is_ext = vector_x < -x_threshold
        return is_ext

    # Neck
    neckAngleMagnitude = angle_with_vertical(shoulder_mid, nose) if nose and nose.get('visibility',0) > min_vis else 0.0
    neckIsExtension = get_extension_flag_from_vertical(shoulder_mid, nose, filming_side, neckAngleMagnitude)

    # Trunk
    trunkAngleMagnitude = angle_with_vertical(hip_mid, shoulder_mid)
    trunkIsExtension = get_extension_flag_from_vertical(hip_mid, shoulder_mid, filming_side, trunkAngleMagnitude)
    trunkRotationAngle = 0.0 # Placeholder - Rotation calc remains approximate

    # Limbs
    results = {}
    for side in ["left", "right"]:
        prefix = "L_" if side == "left" else "R_"
        hip = lm[lm_indices[prefix + "Hip"]]
        shoulder = lm[lm_indices[prefix + "Shoulder"]]
        elbow = lm[lm_indices[prefix + "Elbow"]]
        wrist = lm[lm_indices[prefix + "Wrist"]]
        knee = lm[lm_indices[prefix + "Knee"]]
        ankle = lm[lm_indices[prefix + "Ankle"]]

        # Upper Arm
        ua_angle_mag_rel_trunk = 0.0
        ua_is_ext = False
        if shoulder.get('visibility',0) > min_vis and hip.get('visibility',0) > min_vis and elbow.get('visibility',0) > min_vis:
            trunk_vec_T_2d = vec_subtract_2d(shoulder, hip)
            upper_arm_vec_UA_2d = vec_subtract_2d(elbow, shoulder)
            if trunk_vec_T_2d and upper_arm_vec_UA_2d:
                ua_angle_mag_rel_trunk = angle_between_2d_vectors(trunk_vec_T_2d, upper_arm_vec_UA_2d)
                cross_product_z = vec_cross_2d(trunk_vec_T_2d, upper_arm_vec_UA_2d)
                cross_threshold = 0.001 # Tune this
                if ua_angle_mag_rel_trunk > 10:
                    if filming_side == "left": ua_is_ext = cross_product_z > cross_threshold
                    elif filming_side == "right": ua_is_ext = cross_product_z < -cross_threshold
        results[f"{side}UpperArmAngleMagnitude"] = ua_angle_mag_rel_trunk
        results[f"{side}UpperArmIsExtension"] = ua_is_ext

        # Elbow
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        results[f"{side}ElbowAngle"] = elbow_angle

        # Wrist
        results[f"{side}WristAngle"] = 0.0 # Calculation unreliable

        # Knee
        knee_angle = calculate_angle(hip, knee, ankle)
        results[f"{side}KneeAngle"] = knee_angle

    final_angles = {
        "neckAngleMagnitude": neckAngleMagnitude, "neckIsExtension": neckIsExtension,
        "trunkAngleMagnitude": trunkAngleMagnitude, "trunkIsExtension": trunkIsExtension,
        "trunkRotationAngle": trunkRotationAngle,
        **results
    }
    return final_angles


# -----------------------------
# Revised Scoring Functions
# -----------------------------
def calc_neck_score_revised(angle_magnitude: float, is_extension: bool, rotationFlag: bool, sideBendFlag: bool) -> int:
    # REBA Neck: 0-20 Flexion = 1, >20 Flexion or ANY Extension = 2. +/- Rotation/Bending.
    base = 1
    # Check if extension exists (only if angle is significant)
    if angle_magnitude > 5 and is_extension:
        base = 2
    # Check flexion ranges (only if not extension and angle is significant)
    elif not is_extension and angle_magnitude > 20:
        base = 2
    # If angle <= 5 OR (0-20 flexion AND not extension), base remains 1.
    add = (1 if rotationFlag else 0) + (1 if sideBendFlag else 0)
    return base + add # Max score = 4

def calc_trunk_score_revised(angle_magnitude: float, is_extension: bool, rotationFlag: bool, sideBendFlag: bool) -> int:
    base = 1 # Default to upright
    if angle_magnitude > 5: # Only score deviation if > 5 degrees
        if is_extension:
            base = 2 if angle_magnitude <= 20 else 3 # 0-20 Ext = 2, >20 Ext = 3
        else: # Flexion
            if angle_magnitude <= 20: base = 2      # 5-20 Flex = 2
            elif angle_magnitude <= 60: base = 3    # 20-60 Flex = 3
            else: base = 4                          # >60 Flex = 4
    add = (1 if rotationFlag else 0) + (1 if sideBendFlag else 0)
    return base + add # Max score = 4 + 1 + 1 = 6

def calc_upper_arm_score_revised(angle_relative_to_trunk: float, is_extension: bool, upperArmCorrection: float, shoulderElevation: float, gravityAssist: float) -> int:
    base = 1 # Default to 0-20 Flexion
    angle = angle_relative_to_trunk # Angle (0-180) away from the trunk line
    if is_extension: # Extension relative to trunk
        base = 2 if angle <= 20 else 3 # Ext 0-20 = 2, >20 = 3
    else: # Flexion relative to trunk
        if angle <= 20: base = 1       # Flex 0-20 = 1
        elif angle <= 45: base = 2    # Flex 20-45 = 2
        elif angle <= 90: base = 3    # Flex 45-90 = 3
        else: base = 4                # Flex >90 = 4
    corrected = base + int(upperArmCorrection) + int(shoulderElevation) + int(gravityAssist)
    return max(1, min(6, corrected)) # Clamp 1-6

def calc_leg_score_unified(postureCategory: str, kneeFlexAngle: float) -> int: # Unchanged logic
    if postureCategory == "sittingWalking": return 1
    base = 1 if postureCategory == "standingBoth" else 2
    flex = 180 - (kneeFlexAngle if kneeFlexAngle is not None else 180) # Calculate flexion from internal angle (default straight)
    add = 0
    if 30 <= flex <= 60: add = 1
    elif flex > 60: add = 2
    return base + add

def calc_forearm_score(elbowAngle: float) -> int: # Unchanged logic
    # Assumes elbowAngle is internal angle (0 = straight)
    return 1 if 60 <= elbowAngle <= 100 else 2

def calc_wrist_score(base_score_from_input: int, wristCorrectionFlag: float) -> int: # ★ Uses base score from input ★
    base = base_score_from_input # 1 or 2
    score = base + int(wristCorrectionFlag) # Add +0 or +1
    return max(1, min(3, score)) # Clamp 1-3

def calc_load_score(loadKgInput: float) -> int: # Unchanged logic
    if loadKgInput < 5: return 0
    elif loadKgInput <= 10: return 1
    else: return 2

# -----------------------------
# Lookups & Helpers
# -----------------------------
tableA_Lookup = { "1,1,1":1, "1,1,2":2, "1,1,3":3, "1,1,4":4, "1,2,1":1, "1,2,2":2, "1,2,3":3, "1,2,4":4, "1,3,1":3, "1,3,2":3, "1,3,3":5, "1,3,4":6, "1,4,1":4,"1,4,2":4,"1,4,3":6,"1,4,4":7, "2,1,1":2, "2,1,2":3, "2,1,3":4, "2,1,4":5, "2,2,1":3, "2,2,2":4, "2,2,3":5, "2,2,4":6, "2,3,1":4, "2,3,2":5, "2,3,3":6, "2,3,4":7, "2,4,1":5,"2,4,2":6,"2,4,3":7,"2,4,4":8, "3,1,1":2, "3,1,2":4, "3,1,3":5, "3,1,4":6, "3,2,1":4, "3,2,2":5, "3,2,3":6, "3,2,4":7, "3,3,1":5, "3,3,2":6, "3,3,3":7, "3,3,4":8, "3,4,1":6,"3,4,2":7,"3,4,3":8,"3,4,4":9, "4,1,1":3, "4,1,2":5, "4,1,3":6, "4,1,4":7, "4,2,1":5, "4,2,2":6, "4,2,3":7, "4,2,4":8, "4,3,1":6, "4,3,2":7, "4,3,3":8, "4,3,4":9, "4,4,1":7,"4,4,2":8,"4,4,3":9,"4,4,4":9, "5,1,1":4,"5,1,2":6,"5,1,3":7,"5,1,4":8,"5,2,1":6,"5,2,2":7,"5,2,3":8,"5,2,4":9,"5,3,1":7,"5,3,2":8,"5,3,3":9,"5,3,4":9,"5,4,1":8,"5,4,2":9,"5,4,3":9,"5,4,4":9, "6,1,1":5,"6,1,2":7,"6,1,3":8,"6,1,4":9,"6,2,1":7,"6,2,2":8,"6,2,3":9,"6,2,4":9,"6,3,1":8,"6,3,2":9,"6,3,3":9,"6,3,4":9,"6,4,1":9,"6,4,2":9,"6,4,3":9,"6,4,4":9 }
tableB_Lookup = { "1,1,1":1,"1,1,2":1,"1,1,3":2,"1,2,1":2,"1,2,2":2,"1,2,3":3,"2,1,1":2,"2,1,2":3,"2,1,3":3,"2,2,1":3,"2,2,2":4,"2,2,3":5,"3,1,1":3,"3,1,2":4,"3,1,3":4,"3,2,1":4,"3,2,2":5,"3,2,3":6,"4,1,1":4,"4,1,2":5,"4,1,3":5,"4,2,1":5,"4,2,2":6,"4,2,3":7,"5,1,1":5,"5,1,2":6,"5,1,3":6,"5,2,1":6,"5,2,2":7,"5,2,3":8,"6,1,1":6,"6,1,2":7,"6,1,3":7,"6,2,1":7,"6,2,2":8,"6,2,3":9 }
tableC_Lookup = { "1,1":1,"1,2":1,"1,3":1,"1,4":2,"1,5":3,"1,6":3,"1,7":4,"1,8":5,"1,9":6,"1,10":7,"1,11":7,"1,12":7,"2,1":1,"2,2":2,"2,3":2,"2,4":3,"2,5":4,"2,6":4,"2,7":5,"2,8":6,"2,9":6,"2,10":7,"2,11":7,"2,12":8,"3,1":2,"3,2":3,"3,3":3,"3,4":4,"3,5":5,"3,6":6,"3,7":7,"3,8":7,"3,9":7,"3,10":8,"3,11":8,"3,12":8,"4,1":3,"4,2":4,"4,3":4,"4,4":4,"4,5":5,"4,6":6,"4,7":7,"4,8":8,"4,9":8,"4,10":9,"4,11":9,"4,12":9,"5,1":4,"5,2":4,"5,3":4,"5,4":5,"5,5":6,"5,6":7,"5,7":8,"5,8":8,"5,9":9,"5,10":9,"5,11":9,"5,12":9,"6,1":6,"6,2":6,"6,3":6,"6,4":7,"6,5":8,"6,6":8,"6,7":9,"6,8":9,"6,9":10,"6,10":10,"6,11":10,"6,12":10,"7,1":7,"7,2":7,"7,3":7,"7,4":8,"7,5":9,"7,6":9,"7,7":9,"7,8":10,"7,9":10,"7,10":11,"7,11":11,"7,12":11,"8,1":8,"8,2":8,"8,3":8,"8,4":9,"8,5":10,"8,6":10,"8,7":10,"8,8":10,"8,9":11,"8,10":11,"8,11":11,"8,12":12,"9,1":9,"9,2":9,"9,3":9,"9,4":10,"9,5":10,"9,6":11,"9,7":11,"9,8":11,"9,9":12,"9,10":12,"9,11":12,"9,12":12,"10,1":10,"10,2":10,"10,3":10,"10,4":11,"10,5":11,"10,6":11,"10,7":12,"10,8":12,"10,9":12,"10,10":12,"10,11":12,"10,12":12,"11,1":11,"11,2":11,"11,3":11,"11,4":12,"11,5":12,"11,6":12,"11,7":12,"11,8":12,"11,9":12,"11,10":12,"11,11":12,"11,12":12,"12,1":12,"12,2":12,"12,3":12,"12,4":12,"12,5":12,"12,6":12,"12,7":12,"12,8":12,"12,9":12,"12,10":12,"12,11":12,"12,12":12 }

def lookup_score(table: Dict[str, int], key_parts: List[int], min_vals: List[int], max_vals: List[int]) -> int:
    clamped_keys = [max(min_v, min(max_v, k)) for k, min_v, max_v in zip(key_parts, min_vals, max_vals)]
    key = ",".join(map(str, clamped_keys))
    return table.get(key, 1)

def getScoreA(trunk: int, neck: int, leg: int, loadKgInput: float) -> int:
    base_score = lookup_score(tableA_Lookup, [trunk, neck, leg], [1, 1, 1], [6, 4, 4]) # Max scores from revised funcs
    return base_score + calc_load_score(loadKgInput)

def getScoreB(upperArm: int, forearm: int, wrist: int, coupling: int) -> int:
    base_score = lookup_score(tableB_Lookup, [upperArm, forearm, wrist], [1, 1, 1], [6, 2, 3]) # Max scores from revised funcs
    return base_score + coupling

def getTableCScore(scoreA: int, scoreB: int) -> int:
    # REBA Table C standard range is 1-12 for inputs
    return lookup_score(tableC_Lookup, [scoreA, scoreB], [1, 1], [12, 12])

def get_risk_level(score: int) -> str:
    if score == 1: return "無視できる (Negligible)"
    elif 2 <= score <= 3: return "低リスク (Low)"
    elif 4 <= score <= 7: return "中リスク (Medium)"
    elif 8 <= score <= 10: return "高リスク (High)"
    else: return "非常に高リスク (Very High)" # Score 11-15

# -----------------------------
# Final REBA Score Calculation Function
# -----------------------------
def get_final_reba_score(landmarks: List[Landmark], calib: CalibrationInputs) -> Dict[str, Any]:
    try:
        angles = compute_all_angles(landmarks, calib.filmingSide)
    except HTTPException as e: raise e
    except Exception as e:
        print(f"Error computing angles: {e}")
        raise HTTPException(status_code=500, detail="Error computing body angles.")

    # Flags for scoring adjustments
    neck_twist_flag = calib.neckRotation > 0
    neck_bend_flag = calib.neckLateralBending > 0
    trunk_twist_flag = abs(angles.get("trunkRotationAngle", 0)) >= 5 # Approx rotation check
    trunk_bend_flag = calib.trunkLateralFlexion > 0

    # Calculate component scores using revised functions
    neckScore = calc_neck_score_revised(
        angles.get("neckAngleMagnitude", 0), angles.get("neckIsExtension", False),
        neck_twist_flag, neck_bend_flag
    )
    trunkScore = calc_trunk_score_revised(
        angles.get("trunkAngleMagnitude", 0), angles.get("trunkIsExtension", False),
        trunk_twist_flag, trunk_bend_flag
    )

    # Side-specific scores for limbs
    side = calib.filmingSide
    prefix = "L_" if side == "left" else "R_"

    # Use calculated magnitudes and flags where appropriate
    legScore = calc_leg_score_unified(
        calib.postureCategory, angles.get(f"{side}KneeAngle", 0)
    )
    upperArmScore = calc_upper_arm_score_revised(
        angles.get(f"{side}UpperArmAngleMagnitude", 0), angles.get(f"{side}UpperArmIsExtension", False),
        calib.upperArmCorrection, calib.shoulderElevation, calib.gravityAssist
    )
    forearmScore = calc_forearm_score(
        angles.get(f"{side}ElbowAngle", 0)
    )
    # Use user input for wrist base score
    wristScore = calc_wrist_score(
        calib.wristBaseScore, calib.wristCorrection
    )

    # Combine Scores
    scoreA = getScoreA(trunkScore, neckScore, legScore, calib.loadForce)
    scoreB = getScoreB(upperArmScore, forearmScore, wristScore, calib.coupling)
    tableCScore = getTableCScore(scoreA, scoreB)
    activityScore = calib.staticPosture + calib.repetitiveMovement + calib.unstableMovement
    finalScore = tableCScore + activityScore
    finalScore = max(1, min(15, finalScore)) # REBA final score range 1-15
    riskLevel = get_risk_level(finalScore)

    response_data = {
        "final_score": finalScore,
        "risk_level": riskLevel,
        "computed_angles": angles, # Includes magnitudes and flags
        "intermediate_scores": {
             "neck": neckScore, "trunk": trunkScore, "leg": legScore,
             "upperArm": upperArmScore, "forearm": forearmScore, "wrist": wristScore,
             "scoreA": scoreA, "scoreB": scoreB, "tableC": tableCScore, "activity": activityScore
        }
    }
    print("DEBUG: Returning data:", response_data)
    return response_data

# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/compute_reba")
async def compute_reba_endpoint(input_data: REBAInput):
    try:
        result = get_final_reba_score(input_data.landmarks, input_data.calibInputs)
        return result
    except HTTPException as e: raise e
    except ValidationError as e: raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
         print(f"ERROR: Unexpected error during REBA calculation: {e}")
         import traceback
         traceback.print_exc()
         raise HTTPException(status_code=500, detail=f"Internal server error calculating REBA: {e}")

# -----------------------------
# CORS Middleware
# -----------------------------
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "https://reba-1.onrender.com", # Your specific frontend URL
    # Add any other origins if needed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
async def read_root():
    return {"message": "REBA Evaluation API is running"}
