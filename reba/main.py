# main.py (完全版 - 2025/04/11 時点 - Pydantic V2 Validator 修正済み)

from fastapi import FastAPI, HTTPException
# Pydantic V2 スタイル用に ValidationInfo をインポート
from pydantic import BaseModel, Field, validator, ValidationError, ValidationInfo
from typing import List, Dict, Any, Optional
import math
import traceback # For detailed error logging
from fastapi.middleware.cors import CORSMiddleware # CORS用

app = FastAPI(title="REBA Evaluation API")

# -----------------------------
# モデル定義 (Pydantic V2 対応修正)
# -----------------------------
class Landmark(BaseModel):
    # x, y は必須とする (Optionalを削除)
    x: float
    y: float
    z: Optional[float] = None # Z座標は任意
    visibility: Optional[float] = Field(default=None, ge=0.0, le=1.0) # visibility も任意

    # Pydanticがデフォルトで x, y の存在と型をチェックするため、カスタムバリデータは削除

class CalibrationInputs(BaseModel):
    # --- フィールド定義 ---
    filmingSide: str = Field(..., description="left or right")
    neckRotation: float # Flag: >0 means rotated
    neckLateralBending: float # Flag: >0 means bent
    trunkLateralFlexion: float # Flag: >0 means bent
    loadForce: float # Represents score bracket start (0, 5, 11)
    shockForce: int = Field(..., description="Shock/Rapid force flag (0 or 1)")
    postureCategory: str # "standingBoth", "standingOne", "sittingWalking"
    supportingLeg: Optional[str] = None # Supporting leg ("left" or "right") for standingOne
    upperArmCorrection: float # Abduction/Rotation flag (0 or 1)
    shoulderElevation: float # Shoulder raised flag (0 or 1)
    gravityAssist: float # Arm supported flag (0 or -1)
    wristCorrection: float # Deviation/Twist flag (0 or 1)
    wristBaseScore: int    # Base score from user input (1 or 2)
    staticPosture: int # Activity score flag (0 or 1)
    repetitiveMovement: int # Activity score flag (0 or 1)
    unstableMovement: int # Activity score flag (0 or 1)
    coupling: int # Coupling score addition (0, 1, 2, or 3)

    # --- バリデータ ---
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

    # ★★★ supporting_leg_valid を Pydantic V2 スタイルに修正 ★★★
    @validator('supportingLeg')
    def supporting_leg_valid(cls, v: Optional[str], info: ValidationInfo): # 'values' の代わりに 'info: ValidationInfo'
        # 他のフィールドの値は info.data 経由で取得
        if info.data: # info.data が存在するかチェック
            posture = info.data.get('postureCategory')
            if posture == 'standingOne':
                # 片足立ちの場合、supportingLeg は "left" か "right" でなければならない
                if v not in ['left', 'right']:
                    raise ValueError('Supporting leg ("left" or "right") must be specified for standingOne posture')
            elif v is not None and v not in ['left', 'right']:
                 # 片足立ち以外の場合、None, "left", "right" 以外はエラー
                 raise ValueError('Supporting leg must be "left", "right", or null')
        return v
    # ★★★ ここまで修正 ★★★

    @validator('wristBaseScore')
    def wrist_base_score_valid(cls, v):
        if v not in [1, 2]: raise ValueError('Wrist Base Score must be 1 or 2')
        return v

    @validator('shockForce')
    def shock_force_valid(cls, v):
        if v not in [0, 1]: raise ValueError('Shock Force flag must be 0 or 1')
        return v

class REBAInput(BaseModel):
    landmarks: List[Landmark]
    calibInputs: CalibrationInputs

# -----------------------------
# 2D Vector Calculation Helpers
# -----------------------------
def parse_point(p: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """ Safely return point data if it exists and has valid x, y """
    if p and isinstance(p, dict) and 'x' in p and 'y' in p and \
       isinstance(p['x'], (int, float)) and isinstance(p['y'], (int, float)):
        # Ensure 'visibility' and 'z' are floats if they exist
        p['z'] = float(p['z']) if p.get('z') is not None else 0.0
        p['visibility'] = float(p['visibility']) if p.get('visibility') is not None else None
        return p
    return None

def vec_subtract_2d(p1_data: Dict, p2_data: Dict) -> Optional[Dict[str, float]]:
    """ Subtracts p2 from p1 (p1 - p2) safely for 2D vector """
    p1 = parse_point(p1_data); p2 = parse_point(p2_data)
    if p1 and p2:
        return {"x": p1['x'] - p2['x'], "y": p1['y'] - p2['y']}
    return None

def vec_dot_2d(v1: Dict, v2: Dict) -> float:
    """ Calculates dot product of two 2D vectors """
    return v1.get('x', 0.0) * v2.get('x', 0.0) + v1.get('y', 0.0) * v2.get('y', 0.0)

def vec_cross_2d(v1: Dict, v2: Dict) -> float:
    """ Calculates the scalar value representing the Z-component of the 3D cross product (ax*by - ay*bx) """
    return v1.get('x', 0.0) * v2.get('y', 0.0) - v1.get('y', 0.0) * v2.get('x', 0.0)

def vec_magnitude_2d(v: Optional[Dict[str, float]]) -> float:
    """ Calculates magnitude of a 2D vector """
    if not v: return 0.0
    mag_sq = v.get('x', 0.0)**2 + v.get('y', 0.0)**2
    # Use a slightly larger epsilon to avoid issues with very small non-zero numbers
    return math.sqrt(mag_sq) if mag_sq > 1e-12 else 0.0

def angle_between_2d_vectors(v1_data: Dict, v2_data: Dict) -> float:
    """ Calculate angle between two 2D vectors using dot product (0-180 degrees) """
    v1 = parse_point(v1_data); v2 = parse_point(v2_data) # Use parse_point for safety
    if not v1 or not v2: return 0.0
    mag1 = vec_magnitude_2d(v1); mag2 = vec_magnitude_2d(v2)
    if mag1 * mag2 < 1e-12: return 0.0
    dot = vec_dot_2d(v1, v2)
    # Clamp rigorously
    cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    # Check for potential NaN issues even after clamping due to float precision
    if math.isnan(cos_theta): cos_theta = 1.0 if dot >= 0 else -1.0
    try:
        angle_rad = math.acos(cos_theta)
        return math.degrees(angle_rad)
    except ValueError as e:
         print(f"Warning: Math domain error in acos({cos_theta}): {e}")
         return 0.0 if cos_theta >= 0 else 180.0 # Return 0 or 180 based on sign

def angle_with_vertical(p1_data: Dict, p2_data: Dict, min_visibility: float = 0.5) -> float:
    """ Calculates angle of vector p1->p2 with downward vertical (0-180 deg) """
    p1 = parse_point(p1_data); p2 = parse_point(p2_data)
    # Check visibility using .get with default 0 if key missing
    if not p1 or not p2 or p1.get('visibility', 0.0) < min_visibility or p2.get('visibility', 0.0) < min_visibility:
        return 0.0 # Default to 0 if points invalid or low visibility
    vector = vec_subtract_2d(p2, p1)
    if not vector: return 0.0
    vertical = {"x": 0, "y": 1} # Y increases downwards
    return angle_between_2d_vectors(vector, vertical)

def calculate_angle(p1_data: Dict, p2_data: Dict, p3_data: Dict, min_visibility: float = 0.5) -> float:
    """ Calculates the internal angle at p2 between vectors p2->p1 and p2->p3 (0-180 deg) """
    p1 = parse_point(p1_data); p2 = parse_point(p2_data); p3 = parse_point(p3_data)
    if not p1 or not p2 or not p3 or \
       p1.get('visibility', 1.0) < min_visibility or \
       p2.get('visibility', 1.0) < min_visibility or \
       p3.get('visibility', 1.0) < min_visibility:
        return 0.0 # Return 0 if points invalid or low visibility
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
    min_vis = 0.5 # Visibility threshold

    # --- Midpoints ---
    ls_p = parse_point(lm[lm_indices["L_Shoulder"]]); rs_p = parse_point(lm[lm_indices["R_Shoulder"]])
    lh_p = parse_point(lm[lm_indices["L_Hip"]]); rh_p = parse_point(lm[lm_indices["R_Hip"]])
    shoulder_mid = {"x": (ls_p['x'] + rs_p['x']) / 2, "y": (ls_p['y'] + rs_p['y']) / 2, "visibility": min(ls_p.get('visibility',0), rs_p.get('visibility',0))} if ls_p and rs_p else None
    hip_mid = {"x": (lh_p['x'] + rh_p['x']) / 2, "y": (lh_p['y'] + rh_p['y']) / 2, "visibility": min(lh_p.get('visibility',0), rh_p.get('visibility',0))} if lh_p and rh_p else None
    if not shoulder_mid or not hip_mid or shoulder_mid.get('visibility',0) < min_vis or hip_mid.get('visibility',0) < min_vis:
         raise HTTPException(status_code=400, detail="Cannot compute midpoints (shoulder/hip).")
    nose_p = parse_point(lm[lm_indices["Nose"]])

    # --- Helper: Extension Flag from Vertical + X-Component ---
    def get_extension_flag_from_vertical(p1: Dict, p2: Dict, side: str, angle_mag_vert: float) -> bool:
        is_ext = False
        p1_p = parse_point(p1); p2_p = parse_point(p2)
        if not p1_p or not p2_p or p1_p.get('visibility', 1.0) < min_vis or p2_p.get('visibility', 1.0) < min_vis: return False
        vector_x = p2_p['x'] - p1_p['x']
        x_threshold = 0.02
        if angle_mag_vert > 5: # Only check if not near vertical
            if side == "left": is_ext = vector_x > x_threshold
            elif side == "right": is_ext = vector_x < -x_threshold
        return is_ext

    # --- Neck ---
    neckAngleMagnitude = angle_with_vertical(shoulder_mid, nose_p) if nose_p else 0.0
    neckIsExtension = get_extension_flag_from_vertical(shoulder_mid, nose_p, filming_side, neckAngleMagnitude)

    # --- Trunk ---
    trunkAngleMagnitude = angle_with_vertical(hip_mid, shoulder_mid)
    trunkIsExtension = get_extension_flag_from_vertical(hip_mid, shoulder_mid, filming_side, trunkAngleMagnitude)
    trunkRotationAngle = 0.0 # Placeholder

    # --- Process Limbs ---
    results = {}
    for side in ["left", "right"]:
        prefix = "L_" if side == "left" else "R_"
        hip_p = parse_point(lm[lm_indices[prefix + "Hip"]])
        shoulder_p = parse_point(lm[lm_indices[prefix + "Shoulder"]])
        elbow_p = parse_point(lm[lm_indices[prefix + "Elbow"]])
        wrist_p = parse_point(lm[lm_indices[prefix + "Wrist"]])
        knee_p = parse_point(lm[lm_indices[prefix + "Knee"]])
        ankle_p = parse_point(lm[lm_indices[prefix + "Ankle"]])

        # Upper Arm (2D Relative Angle & Direction)
        ua_angle_mag_rel_trunk = 0.0
        ua_is_ext = False
        if shoulder_p and hip_p and elbow_p and shoulder_p.get('visibility',0) > min_vis and hip_p.get('visibility',0) > min_vis and elbow_p.get('visibility',0) > min_vis:
            trunk_vec_T_2d = vec_subtract_2d(shoulder_p, hip_p)
            upper_arm_vec_UA_2d = vec_subtract_2d(elbow_p, shoulder_p)
            if trunk_vec_T_2d and upper_arm_vec_UA_2d:
                ua_angle_mag_rel_trunk = angle_between_2d_vectors(trunk_vec_T_2d, upper_arm_vec_UA_2d)
                cross_product_z = vec_cross_2d(trunk_vec_T_2d, upper_arm_vec_UA_2d)
                cross_threshold = 0.001 # Tune this
                if ua_angle_mag_rel_trunk > 10:
                    # Sign interpretation needs validation based on coordinate system/data
                    if filming_side == "left": ua_is_ext = cross_product_z > cross_threshold # Example: Positive cross product = Extension?
                    elif filming_side == "right": ua_is_ext = cross_product_z < -cross_threshold # Example: Negative cross product = Extension?
        results[f"{side}UpperArmAngleMagnitude"] = ua_angle_mag_rel_trunk
        results[f"{side}UpperArmIsExtension"] = ua_is_ext

        # Elbow (Internal Angle Magnitude)
        elbow_angle = calculate_angle(shoulder_p, elbow_p, wrist_p)
        results[f"{side}ElbowAngle"] = elbow_angle

        # Wrist (Unreliable)
        results[f"{side}WristAngle"] = 0.0

        # Knee (Internal Angle Magnitude)
        knee_angle = calculate_angle(hip_p, knee_p, ankle_p)
        results[f"{side}KneeAngle"] = knee_angle

    # Combine results
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
    base = 1
    if angle_magnitude > 5 and is_extension: base = 2
    elif not is_extension and angle_magnitude > 20: base = 2
    add = (1 if rotationFlag else 0) + (1 if sideBendFlag else 0)
    return base + add

def calc_trunk_score_revised(angle_magnitude: float, is_extension: bool, rotationFlag: bool, sideBendFlag: bool) -> int:
    base = 1
    if angle_magnitude > 5:
        if is_extension: base = 2 if angle_magnitude <= 20 else 3
        else: # Flexion
            if angle_magnitude <= 20: base = 2
            elif angle_magnitude <= 60: base = 3
            else: base = 4
    add = (1 if rotationFlag else 0) + (1 if sideBendFlag else 0)
    return base + add

def calc_upper_arm_score_revised(angle_relative_to_trunk: float, is_extension: bool, upperArmCorrection: float, shoulderElevation: float, gravityAssist: float) -> int:
    base = 1
    angle = angle_relative_to_trunk
    if is_extension: base = 2 if angle <= 20 else 3
    else: # Flexion or neutral
        if angle <= 20: base = 1
        elif angle <= 45: base = 2
        elif angle <= 90: base = 3
        else: base = 4
    corrected = base + int(upperArmCorrection) + int(shoulderElevation) + int(gravityAssist)
    return max(1, min(6, corrected))

def calc_leg_score_unified(postureCategory: str, kneeFlexAngle: Optional[float]) -> int:
    if postureCategory == "sittingWalking": return 1
    base = 1 if postureCategory == "standingBoth" else 2
    internal_angle = 180.0 if kneeFlexAngle is None else kneeFlexAngle
    flex = 180.0 - internal_angle
    add = 0
    if 30 <= flex <= 60: add = 1
    elif flex > 60: add = 2
    return base + add

def calc_forearm_score(elbowAngle: Optional[float]) -> int:
    internal_angle = 90.0 if elbowAngle is None else elbowAngle
    # Use correct internal angle range for 60-100 flexion
    return 1 if 80 <= internal_angle <= 120 else 2

def calc_wrist_score(base_score_from_input: int, wristCorrectionFlag: float) -> int: # Uses base score from input
    base = base_score_from_input
    score = base + int(wristCorrectionFlag)
    return max(1, min(3, score))

def calc_load_score(loadKgInput: float, shockForceFlag: int) -> int: # Added shockForceFlag
    base_score = 0
    if loadKgInput < 5: base_score = 0
    elif loadKgInput <= 10: base_score = 1
    else: base_score = 2
    final_load_score = base_score + int(shockForceFlag)
    return max(0, min(3, final_load_score)) # Clamp 0-3

# -----------------------------
# Lookups & Helpers
# -----------------------------
# Ensure these tables are complete and correct based on REBA standard
tableA_Lookup = { "1,1,1":1, "1,1,2":2, "1,1,3":3, "1,1,4":4, "1,2,1":1, "1,2,2":2, "1,2,3":3, "1,2,4":4, "1,3,1":3, "1,3,2":3, "1,3,3":5, "1,3,4":6, "1,4,1":4,"1,4,2":4,"1,4,3":6,"1,4,4":7, "2,1,1":2, "2,1,2":3, "2,1,3":4, "2,1,4":5, "2,2,1":3, "2,2,2":4, "2,2,3":5, "2,2,4":6, "2,3,1":4, "2,3,2":5, "2,3,3":6, "2,3,4":7, "2,4,1":5,"2,4,2":6,"2,4,3":7,"2,4,4":8, "3,1,1":2, "3,1,2":4, "3,1,3":5, "3,1,4":6, "3,2,1":4, "3,2,2":5, "3,2,3":6, "3,2,4":7, "3,3,1":5, "3,3,2":6, "3,3,3":7, "3,3,4":8, "3,4,1":6,"3,4,2":7,"3,4,3":8,"3,4,4":9, "4,1,1":3, "4,1,2":5, "4,1,3":6, "4,1,4":7, "4,2,1":5, "4,2,2":6, "4,2,3":7, "4,2,4":8, "4,3,1":6, "4,3,2":7, "4,3,3":8, "4,3,4":9, "4,4,1":7,"4,4,2":8,"4,4,3":9,"4,4,4":9, "5,1,1":4,"5,1,2":6,"5,1,3":7,"5,1,4":8,"5,2,1":6,"5,2,2":7,"5,2,3":8,"5,2,4":9,"5,3,1":7,"5,3,2":8,"5,3,3":9,"5,3,4":9,"5,4,1":8,"5,4,2":9,"5,4,3":9,"5,4,4":9, "6,1,1":5,"6,1,2":7,"6,1,3":8,"6,1,4":9,"6,2,1":7,"6,2,2":8,"6,2,3":9,"6,2,4":9,"6,3,1":8,"6,3,2":9,"6,3,3":9,"6,3,4":9,"6,4,1":9,"6,4,2":9,"6,4,3":9,"6,4,4":9 }
tableB_Lookup = { "1,1,1":1,"1,1,2":1,"1,1,3":2,"1,2,1":2,"1,2,2":2,"1,2,3":3,"2,1,1":2,"2,1,2":3,"2,1,3":3,"2,2,1":3,"2,2,2":4,"2,2,3":5,"3,1,1":3,"3,1,2":4,"3,1,3":4,"3,2,1":4,"3,2,2":5,"3,2,3":6,"4,1,1":4,"4,1,2":5,"4,1,3":5,"4,2,1":5,"4,2,2":6,"4,2,3":7,"5,1,1":5,"5,1,2":6,"5,1,3":6,"5,2,1":6,"5,2,2":7,"5,2,3":8,"6,1,1":6,"6,1,2":7,"6,1,3":7,"6,2,1":7,"6,2,2":8,"6,2,3":9 }
tableC_Lookup = { "1,1":1,"1,2":1,"1,3":1,"1,4":2,"1,5":3,"1,6":3,"1,7":4,"1,8":5,"1,9":6,"1,10":7,"1,11":7,"1,12":7,"2,1":1,"2,2":2,"2,3":2,"2,4":3,"2,5":4,"2,6":4,"2,7":5,"2,8":6,"2,9":6,"2,10":7,"2,11":7,"2,12":8,"3,1":2,"3,2":3,"3,3":3,"3,4":4,"3,5":5,"3,6":6,"3,7":7,"3,8":7,"3,9":7,"3,10":8,"3,11":8,"3,12":8,"4,1":3,"4,2":4,"4,3":4,"4,4":4,"4,5":5,"4,6":6,"4,7":7,"4,8":8,"4,9":8,"4,10":9,"4,11":9,"4,12":9,"5,1":4,"5,2":4,"5,3":4,"5,4":5,"5,5":6,"5,6":7,"5,7":8,"5,8":8,"5,9":9,"5,10":9,"5,11":9,"5,12":9,"6,1":6,"6,2":6,"6,3":6,"6,4":7,"6,5":8,"6,6":8,"6,7":9,"6,8":9,"6,9":10,"6,10":10,"6,11":10,"6,12":10,"7,1":7,"7,2":7,"7,3":7,"7,4":8,"7,5":9,"7,6":9,"7,7":9,"7,8":10,"7,9":10,"7,10":11,"7,11":11,"7,12":11,"8,1":8,"8,2":8,"8,3":8,"8,4":9,"8,5":10,"8,6":10,"8,7":10,"8,8":10,"8,9":11,"8,10":11,"8,11":11,"8,12":12,"9,1":9,"9,2":9,"9,3":9,"9,4":10,"9,5":10,"9,6":11,"9,7":11,"9,8":11,"9,9":12,"9,10":12,"9,11":12,"9,12":12,"10,1":10,"10,2":10,"10,3":10,"10,4":11,"10,5":11,"10,6":11,"10,7":12,"10,8":12,"10,9":12,"10,10":12,"10,11":12,"10,12":12,"11,1":11,"11,2":11,"11,3":11,"11,4":12,"11,5":12,"11,6":12,"11,7":12,"11,8":12,"11,9":12,"11,10":12,"11,11":12,"11,12":12,"12,1":12,"12,2":12,"12,3":12,"12,4":12,"12,5":12,"12,6":12,"12,7":12,"12,8":12,"12,9":12,"12,10":12,"12,11":12,"12,12":12 }

def lookup_score(table: Dict[str, int], key_parts: List[int], min_vals: List[int], max_vals: List[int]) -> int:
    clamped_keys = [max(min_v, min(max_v, int(round(k)))) for k, min_v, max_v in zip(key_parts, min_vals, max_vals)]
    key = ",".join(map(str, clamped_keys))
    return table.get(key, 1)

def getScoreA(trunk: int, neck: int, leg: int, loadKgInput: float, shockForceFlag: int) -> int: # Added shockForceFlag
    table_a_score = lookup_score(tableA_Lookup, [trunk, neck, leg], [1, 1, 1], [6, 4, 4])
    final_load_score = calc_load_score(loadKgInput, shockForceFlag)
    return table_a_score + final_load_score

def getScoreB(upperArm: int, forearm: int, wrist: int, coupling: int) -> int:
    base_score = lookup_score(tableB_Lookup, [upperArm, forearm, wrist], [1, 1, 1], [6, 2, 3])
    return base_score + coupling

def getTableCScore(scoreA: int, scoreB: int) -> int:
    # Standard REBA Table C range is 1-12 for inputs
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
    """ Calculates the final REBA score and intermediate values """
    try:
        angles = compute_all_angles(landmarks, calib.filmingSide)
    except HTTPException as e: raise e
    except Exception as e:
        print(f"ERROR during angle computation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Angle computation failed: {e}")

    try:
        # --- Flags ---
        neck_twist_flag = calib.neckRotation > 0
        neck_bend_flag = calib.neckLateralBending > 0
        trunk_twist_flag = abs(angles.get("trunkRotationAngle", 0)) >= 5
        trunk_bend_flag = calib.trunkLateralFlexion > 0

        # --- Component Scores ---
        neckScore = calc_neck_score_revised( angles.get("neckAngleMagnitude", 0), angles.get("neckIsExtension", False), neck_twist_flag, neck_bend_flag )
        trunkScore = calc_trunk_score_revised( angles.get("trunkAngleMagnitude", 0), angles.get("trunkIsExtension", False), trunk_twist_flag, trunk_bend_flag )

        # --- Leg Score ---
        left_knee_angle = angles.get("leftKneeAngle"); right_knee_angle = angles.get("rightKneeAngle")
        legScore = 1
        if calib.postureCategory == "sittingWalking": legScore = 1
        elif calib.postureCategory == "standingBoth":
            score_left = calc_leg_score_unified(calib.postureCategory, left_knee_angle)
            score_right = calc_leg_score_unified(calib.postureCategory, right_knee_angle)
            legScore = max(score_left, score_right)
        elif calib.postureCategory == "standingOne":
            knee_angle_to_use = None
            if calib.supportingLeg == "left": knee_angle_to_use = left_knee_angle
            elif calib.supportingLeg == "right": knee_angle_to_use = right_knee_angle
            if knee_angle_to_use is not None: legScore = calc_leg_score_unified(calib.postureCategory, knee_angle_to_use)
            else: # Fallback
                print(f"DEBUG: StandingOne - Supporting leg ('{calib.supportingLeg}') angle invalid or missing! Using max score.")
                score_left = calc_leg_score_unified(calib.postureCategory, left_knee_angle)
                score_right = calc_leg_score_unified(calib.postureCategory, right_knee_angle)
                legScore = max(score_left, score_right)

        # --- Limb Scores ---
        side_to_eval_arm = calib.filmingSide
        prefix = "L_" if side_to_eval_arm == "left" else "R_"
        upperArmScore = calc_upper_arm_score_revised( angles.get(f"{side_to_eval_arm}UpperArmAngleMagnitude", 0), angles.get(f"{side_to_eval_arm}UpperArmIsExtension", False), calib.upperArmCorrection, calib.shoulderElevation, calib.gravityAssist )
        forearmScore = calc_forearm_score( angles.get(f"{side_to_eval_arm}ElbowAngle") )
        wristScore = calc_wrist_score( calib.wristBaseScore, calib.wristCorrection )

        # --- Combine Scores ---
        scoreA = getScoreA(trunkScore, neckScore, legScore, calib.loadForce, calib.shockForce) # Pass shockForce
        scoreB = getScoreB(upperArmScore, forearmScore, wristScore, calib.coupling)
        tableCScore = getTableCScore(scoreA, scoreB)
        activityScore = calib.staticPosture + calib.repetitiveMovement + calib.unstableMovement
        finalScore = tableCScore + activityScore
        finalScore = max(1, min(15, finalScore)) # Clamp 1-15
        riskLevel = get_risk_level(finalScore)

        response_data = {
            "final_score": finalScore, "risk_level": riskLevel,
            "computed_angles": angles,
            "intermediate_scores": {
                 "neck": neckScore, "trunk": trunkScore, "leg": legScore,
                 "upperArm": upperArmScore, "forearm": forearmScore, "wrist": wristScore,
                 "scoreA": scoreA, "scoreB": scoreB, "tableC": tableCScore, "activity": activityScore
            }
        }
        print("DEBUG: Returning data:", response_data) # Keep debug log
        return response_data

    except Exception as e:
         print(f"ERROR: Unexpected error during REBA score calculation logic: {e}")
         traceback.print_exc()
         raise HTTPException(status_code=500, detail=f"Score calculation failed: {e}")

# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/compute_reba")
async def compute_reba_endpoint(input_data: REBAInput):
    try:
        # Basic landmark count check
        if not input_data.landmarks or len(input_data.landmarks) < lm_indices["R_Ankle"] + 1: # Use max index needed
             raise HTTPException(status_code=400, detail=f"Insufficient landmarks provided ({len(input_data.landmarks)}).")
        # Pydantic validation runs implicitly by type hint

        result = get_final_reba_score(input_data.landmarks, input_data.calibInputs)
        return result
    except HTTPException as e: raise e # Propagate known errors
    except ValidationError as e: raise HTTPException(status_code=422, detail=e.errors()) # Pydantic errors
    except Exception as e: # Catch-all for other unexpected errors
         print(f"ERROR: Unhandled exception in /compute_reba endpoint: {e}")
         traceback.print_exc()
         raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# -----------------------------
# CORS Middleware
# -----------------------------
# ★★★ ご自身のフロントエンドURLを正しく設定してください ★★★
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "https://reba-1.onrender.com", # 例: ユーザー提供のURL
    # 必要に応じて他のオリジンを追加
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

# -----------------------------
# Optional: Uvicorn runner for local testing
# -----------------------------
# if __name__ == "__main__":
#     import uvicorn
#     print("Starting Uvicorn server locally on http://127.0.0.1:8000")
#     # Use reload=True for development
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
