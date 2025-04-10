# main.py (純粋な2D計算 + 体幹基準の上腕角度評価 最終版)

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

class REBAInput(BaseModel):
    landmarks: List[Landmark]
    calibInputs: CalibrationInputs

# -----------------------------
# 2D Vector Calculation Helpers
# -----------------------------
def parse_point(p: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """ Safely return point data if it exists """
    return p if p and isinstance(p, dict) else None

def vec_subtract_2d(p1_data: Dict, p2_data: Dict) -> Optional[Dict[str, float]]:
    """ Subtracts p2 from p1 (p1 - p2) safely for 2D vector """
    p1 = parse_point(p1_data)
    p2 = parse_point(p2_data)
    if p1 and p2 and 'x' in p1 and 'y' in p1 and 'x' in p2 and 'y' in p2:
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
    return math.sqrt(mag_sq) if mag_sq > 1e-9 else 0.0

def angle_between_2d_vectors(v1_data: Dict, v2_data: Dict) -> float:
    """ Calculate angle between two 2D vectors using dot product (0-180 degrees) """
    v1 = parse_point(v1_data) # Reuse parse_point as vectors are dicts
    v2 = parse_point(v2_data)
    if not v1 or not v2: return 0.0
    mag1 = vec_magnitude_2d(v1)
    mag2 = vec_magnitude_2d(v2)
    if mag1 * mag2 < 1e-9: return 0.0 # Avoid division by zero

    dot = vec_dot_2d(v1, v2)
    cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))

    if math.isnan(cos_theta): cos_theta = 1.0 if dot > 0 else -1.0
    try:
        angle_rad = math.acos(cos_theta)
        return math.degrees(angle_rad)
    except ValueError as e:
         print(f"Error in acos({cos_theta}): {e}")
         return 0.0

def angle_with_vertical(p1_data: Dict, p2_data: Dict, min_visibility: float = 0.5) -> float:
    """ Calculates angle of vector p1->p2 with downward vertical (0-180 deg) """
    p1 = parse_point(p1_data)
    p2 = parse_point(p2_data)
    if not p1 or not p2 or p1.get('visibility', 1.0) < min_visibility or p2.get('visibility', 1.0) < min_visibility:
        return 0.0 # Default to 0 if points invalid or low visibility

    vector = vec_subtract_2d(p2, p1)
    if not vector: return 0.0
    vertical = {"x": 0, "y": 1} # Y increases downwards

    return angle_between_2d_vectors(vector, vertical) # Use the general function

def calculate_angle(p1_data: Dict, p2_data: Dict, p3_data: Dict, min_visibility: float = 0.5) -> float:
    """ Calculates the internal angle at p2 between vectors p2->p1 and p2->p3 (0-180 deg) """
    p1 = parse_point(p1_data); p2 = parse_point(p2_data); p3 = parse_point(p3_data)
    if not p1 or not p2 or not p3 or \
       p1.get('visibility', 1.0) < min_visibility or \
       p2.get('visibility', 1.0) < min_visibility or \
       p3.get('visibility', 1.0) < min_visibility:
        return 0.0 # Return 0 if points invalid or low visibility

    v1 = vec_subtract_2d(p1, p2) # Vector p2->p1
    v2 = vec_subtract_2d(p3, p2) # Vector p2->p3
    if not v1 or not v2: return 0.0

    return angle_between_2d_vectors(v1, v2)

# -----------------------------
# ランドマークから角度計算 (Pure 2D)
# -----------------------------
def compute_all_angles(landmarks: List[Landmark], filming_side: str) -> Dict[str, Any]:
    lm_indices = { "Nose": 0, "L_Shoulder": 11, "R_Shoulder": 12, "L_Elbow": 13, "R_Elbow": 14, "L_Wrist": 15, "R_Wrist": 16, "L_Hip": 23, "R_Hip": 24, "L_Knee": 25, "R_Knee": 26, "L_Ankle": 27, "R_Ankle": 28 }
    max_index = max(lm_indices.values())
    if len(landmarks) <= max_index: raise HTTPException(status_code=400, detail=f"Not enough landmarks ({len(landmarks)}). Need {max_index + 1}")

    lm = [lm.model_dump() if hasattr(lm, 'model_dump') else lm.dict() for lm in landmarks]
    min_vis = 0.5

    # --- Midpoints ---
    ls = lm[lm_indices["L_Shoulder"]]; rs = lm[lm_indices["R_Shoulder"]]
    lh = lm[lm_indices["L_Hip"]]; rh = lm[lm_indices["R_Hip"]]
    shoulder_mid = {"x": (ls['x'] + rs['x']) / 2, "y": (ls['y'] + rs['y']) / 2, "visibility": min(ls.get('visibility',0), rs.get('visibility',0))} if ls and rs else None
    hip_mid = {"x": (lh['x'] + rh['x']) / 2, "y": (lh['y'] + rh['y']) / 2, "visibility": min(lh.get('visibility',0), rh.get('visibility',0))} if lh and rh else None
    if not shoulder_mid or not hip_mid or shoulder_mid.get('visibility',0) < min_vis or hip_mid.get('visibility',0) < min_vis:
         raise HTTPException(status_code=400, detail="Cannot compute midpoints (shoulder/hip).")
    nose = lm[lm_indices["Nose"]]

    # --- Helper: Extension Flag from Vertical + X-Component ---
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

    # --- Neck (Magnitude vs Vertical, Direction from X vs Vertical) ---
    neckAngleMagnitude = angle_with_vertical(shoulder_mid, nose) if nose and nose.get('visibility',0) > min_vis else 0.0
    neckIsExtension = get_extension_flag_from_vertical(shoulder_mid, nose, filming_side, neckAngleMagnitude)

    # --- Trunk (Magnitude vs Vertical, Direction from X vs Vertical) ---
    trunkAngleMagnitude = angle_with_vertical(hip_mid, shoulder_mid)
    trunkIsExtension = get_extension_flag_from_vertical(hip_mid, shoulder_mid, filming_side, trunkAngleMagnitude)
    trunkRotationAngle = 0.0 # Placeholder - Rotation calculation needs review, remains approximate in 2D

    # --- Process Limbs ---
    results = {}
    for side in ["left", "right"]:
        prefix = "L_" if side == "left" else "R_"
        hip = lm[lm_indices[prefix + "Hip"]]
        shoulder = lm[lm_indices[prefix + "Shoulder"]]
        elbow = lm[lm_indices[prefix + "Elbow"]]
        wrist = lm[lm_indices[prefix + "Wrist"]]
        knee = lm[lm_indices[prefix + "Knee"]]
        ankle = lm[lm_indices[prefix + "Ankle"]]

        ua_angle_mag_rel_trunk = 0.0
        ua_is_ext = False
        # --- ★ Upper Arm (2D Relative Angle & Direction using 2D Cross Product) ★ ---
        if shoulder.get('visibility',0) > min_vis and hip.get('visibility',0) > min_vis and elbow.get('visibility',0) > min_vis:
            trunk_vec_T_2d = vec_subtract_2d(shoulder, hip)
            upper_arm_vec_UA_2d = vec_subtract_2d(elbow, shoulder)

            if trunk_vec_T_2d and upper_arm_vec_UA_2d: # Ensure vectors are valid
                # Magnitude: Angle between Trunk and Upper Arm in 2D
                ua_angle_mag_rel_trunk = angle_between_2d_vectors(trunk_vec_T_2d, upper_arm_vec_UA_2d)

                # Direction: Use 2D Cross Product T x UA sign and filming_side
                cross_product_z = vec_cross_2d(trunk_vec_T_2d, upper_arm_vec_UA_2d)
                # Threshold to ignore small values near alignment (magnitude depends on vector lengths)
                # Using a small fixed threshold for the scalar result
                cross_threshold = 0.001 # Tune this value based on testing

                if ua_angle_mag_rel_trunk > 10: # Check if significantly deviated
                    # Assuming X right, Y down: T=(Tx, Ty<0). UA=(UAx, UAy). Crs = Tx*UAy - Ty*UAx
                    if filming_side == "left":
                        # Left view: Extension=Screen Right. UA is clockwise from T -> Crs > threshold?
                        if cross_product_z > cross_threshold: ua_is_ext = True
                    elif filming_side == "right":
                        # Right view: Extension=Screen Left. UA is counter-clockwise from T -> Crs < -threshold?
                        if cross_product_z < -cross_threshold: ua_is_ext = True
                    # This sign interpretation needs validation!

        results[f"{side}UpperArmAngleMagnitude"] = ua_angle_mag_rel_trunk # Angle relative to trunk (2D)
        results[f"{side}UpperArmIsExtension"] = ua_is_ext               # Direction relative to trunk (2D)

        # --- Elbow (Internal Angle Magnitude) ---
        elbow_angle = calculate_angle(shoulder, elbow, wrist) # Uses 2D dot product implicitly
        results[f"{side}ElbowAngle"] = elbow_angle

        # --- Wrist (Unreliable) ---
        results[f"{side}WristAngle"] = 0.0

        # --- Knee (Internal Angle Magnitude) ---
        knee_angle = calculate_angle(hip, knee, ankle) # Uses 2D dot product implicitly
        results[f"{side}KneeAngle"] = knee_angle

    # Combine results
    final_angles = {
        "neckAngleMagnitude": neckAngleMagnitude, "neckIsExtension": neckIsExtension,
        "trunkAngleMagnitude": trunkAngleMagnitude, "trunkIsExtension": trunkIsExtension,
        "trunkRotationAngle": trunkRotationAngle, # Still approximate
        **results
    }
    return final_angles


# -----------------------------
# ★★★ Revised Scoring Functions ★★★
# -----------------------------
def calc_neck_score_revised(angle_magnitude: float, is_extension: bool, rotationFlag: bool, sideBendFlag: bool) -> int:
    # REBA Neck: 0-20 Flexion = 1, >20 Flexion or ANY Extension = 2. +/- Rotation/Bending.
    base = 0
    if angle_magnitude <= 5: # Consider near upright as neutral
        base = 1
    elif is_extension: # Any extension > 5 degrees from vertical
        base = 2
    else: # Flexion (> 5 degrees)
        if angle_magnitude <= 20: base = 1 # 0-20 Flexion
        else: base = 2 # > 20 Flexion
    add = (1 if rotationFlag else 0) + (1 if sideBendFlag else 0)
    return base + add # Max score = 2 + 1 + 1 = 4

def calc_trunk_score_revised(angle_magnitude: float, is_extension: bool, rotationFlag: bool, sideBendFlag: bool) -> int:
    base = 0
    if angle_magnitude <= 5: base = 1 # Upright
    elif is_extension:       # Extension
        if angle_magnitude <= 20: base = 2 # 0-20 Ext
        else: base = 3                     # >20 Ext
    else:                    # Flexion
        if angle_magnitude <= 20: base = 2 # 0-20 Flex (REBA rule starts at >0 or >5?) Let's assume >5
        elif angle_magnitude <= 60: base = 3 # 20-60 Flex
        else: base = 4                     # >60 Flex
    add = (1 if rotationFlag else 0) + (1 if sideBendFlag else 0)
    return base + add # Max score = 4 + 1 + 1 = 6? Check REBA sheet for combined adds

def calc_upper_arm_score_revised(angle_relative_to_trunk: float, is_extension: bool, upperArmCorrection: float, shoulderElevation: float, gravityAssist: float) -> int:
    base = 0
    angle = angle_relative_to_trunk # Angle (0-180) away from the trunk line

    if is_extension: # Extension relative to trunk
        # REBA Extension: 0-20 deg = 2, >20 deg = 3
        if angle <= 20: base = 2
        else: base = 3
    else: # Flexion (or neutral/aligned) relative to trunk
        # REBA Flexion: 0-20 deg = 1, 20-45 = 2, 45-90 = 3, >90 = 4
        if angle <= 20: base = 1
        elif angle <= 45: base = 2
        elif angle <= 90: base = 3
        else: base = 4

    # Corrections: +1 Abd/Rot, +1 Raised, -1 Supported
    corrected = base + int(upperArmCorrection) + int(shoulderElevation) + int(gravityAssist)
    return max(1, min(6, corrected)) # Clamp 1-6

# --- Original scoring functions for Leg, Forearm, Wrist, Load ---
# (These only depend on angle magnitude or direct input)
def calc_leg_score_unified(postureCategory: str, kneeFlexAngle: float) -> int: # Uses knee angle magnitude
    if postureCategory == "sittingWalking": return 1
    base = 1 if postureCategory == "standingBoth" else 2
    flex = 180 - kneeFlexAngle # Approximate flexion from internal angle
    add = 0
    if 30 <= flex <= 60: add = 1
    elif flex > 60: add = 2
    return base + add

def calc_forearm_score(elbowAngle: float) -> int: # Uses elbow angle magnitude
    return 1 if 60 <= elbowAngle <= 100 else 2

def calc_wrist_score(wristAngle: float, wristCorrection: float) -> int: # Uses magnitude (currently 0) + flag
    base = 1 # Defaulting to base 1 as calc is unreliable
    score = base + int(wristCorrection)
    return max(1, min(3, score)) # Check REBA table for exact max score (Table B suggests 3 is max base?)

def calc_load_score(loadKgInput: float) -> int: # Uses direct input
    if loadKgInput < 5: return 0
    elif loadKgInput <= 10: return 1
    else: return 2

# --- ルックアップテーブル & ヘルパー (Ensure ranges match revised scores) ---
tableA_Lookup = { ... } # Max Trunk=6, Max Neck=4, Max Leg=4. Check keys.
tableB_Lookup = { ... } # Max UA=6, Max FA=2, Max Wrist=3. Check keys.
tableC_Lookup = { ... } # Check ranges.

def lookup_score(table: Dict[str, int], key_parts: List[int], min_vals: List[int], max_vals: List[int]) -> int:
    clamped_keys = [max(min_v, min(max_v, k)) for k, min_v, max_v in zip(key_parts, min_vals, max_vals)]
    key = ",".join(map(str, clamped_keys))
    return table.get(key, 1) # Default to 1 if key somehow still invalid

def getScoreA(trunk: int, neck: int, leg: int, loadKgInput: float) -> int:
    base_score = lookup_score(tableA_Lookup, [trunk, neck, leg], [1, 1, 1], [6, 4, 4]) # Update max vals
    return base_score + calc_load_score(loadKgInput)

def getScoreB(upperArm: int, forearm: int, wrist: int, coupling: int) -> int:
    base_score = lookup_score(tableB_Lookup, [upperArm, forearm, wrist], [1, 1, 1], [6, 2, 3]) # Update max vals
    return base_score + coupling # coupling score is 0, 1, 2, or 3

def getTableCScore(scoreA: int, scoreB: int) -> int:
    # Max Score A = lookup(6,4,4)+load(2) -> Lookup table max is 9, so 9+2=11?
    # Max Score B = lookup(6,2,3)+coupling(3) -> Lookup table max is 9, so 9+3=12?
    # Standard REBA Table C goes up to 12 for both A and B scores.
    return lookup_score(tableC_Lookup, [scoreA, scoreB], [1, 1], [12, 12])

def get_risk_level(score: int) -> str: # Unchanged
    # ...

# --- ★★★ 修正版: get_final_reba_score ★★★ ---
def get_final_reba_score(landmarks: List[Landmark], calib: CalibrationInputs) -> Dict[str, Any]:
    try:
        # Pass filming_side to angle calculation
        angles = compute_all_angles(landmarks, calib.filmingSide)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error computing angles: {e}")
        raise HTTPException(status_code=500, detail="Error computing body angles.")

    # --- Get Flags for Scoring ---
    neck_twist_flag = calib.neckRotation > 0
    neck_bend_flag = calib.neckLateralBending > 0
    # Use calculated trunk rotation angle with a threshold
    trunk_twist_flag = abs(angles.get("trunkRotationAngle", 0)) >= 5 # Threshold for twist
    trunk_bend_flag = calib.trunkLateralFlexion > 0

    # --- Calculate Scores using REVISED functions ---
    neckScore = calc_neck_score_revised( # Use revised neck score
        angles.get("neckAngleMagnitude", 0), angles.get("neckIsExtension", False),
        neck_twist_flag, neck_bend_flag
    )
    trunkScore = calc_trunk_score_revised( # Use revised trunk score
        angles.get("trunkAngleMagnitude", 0), angles.get("trunkIsExtension", False),
        trunk_twist_flag, trunk_bend_flag
    )

    # Side-specific scores for limbs
    side = calib.filmingSide
    prefix = "L_" if side == "left" else "R_"

    legScore = calc_leg_score_unified( # Uses knee angle magnitude
        calib.postureCategory, angles.get(f"{side}KneeAngle", 0) # Pass correct side's knee angle
    )
    upperArmScore = calc_upper_arm_score_revised( # Use revised upper arm score
        angles.get(f"{side}UpperArmAngleMagnitude", 0), angles.get(f"{side}UpperArmIsExtension", False),
        calib.upperArmCorrection, calib.shoulderElevation, calib.gravityAssist
    )
    forearmScore = calc_forearm_score( # Uses elbow angle magnitude
        angles.get(f"{side}ElbowAngle", 0)
    )
    wristScore = calc_wrist_score( # Uses magnitude (0) + flag
        angles.get(f"{side}WristAngle", 0), calib.wristCorrection
    )

    # --- Combine Scores ---
    scoreA = getScoreA(trunkScore, neckScore, legScore, calib.loadForce)
    scoreB = getScoreB(upperArmScore, forearmScore, wristScore, calib.coupling)
    tableCScore = getTableCScore(scoreA, scoreB)
    activityScore = calib.staticPosture + calib.repetitiveMovement + calib.unstableMovement
    finalScore = tableCScore + activityScore
    finalScore = max(1, min(15, finalScore)) # Clamp 1-15
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
    print("DEBUG: Returning data:", response_data) # Debug log
    return response_data

# -----------------------------
# API エンドポイント (変更なし)
# -----------------------------
@app.post("/compute_reba")
async def compute_reba_endpoint(input_data: REBAInput):
    try:
        result = get_final_reba_score(input_data.landmarks, input_data.calibInputs)
        return result
    except HTTPException as e: raise e
    except ValidationError as e: raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e: # ... (エラーハンドリング) ...

# -----------------------------
# CORS ミドルウェア設定 (変更なし)
# -----------------------------
origins = [ /* ... include frontend URL ... */ ]
app.add_middleware( CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["POST", "GET"], allow_headers=["*"], )

# -----------------------------
# ルートエンドポイント (変更なし)
# -----------------------------
@app.get("/")
async def read_root(): return {"message": "REBA Evaluation API is running"}
