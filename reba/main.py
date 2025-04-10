# main.py (最終版 - 衝撃力追加)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Dict, Any, Optional
import math
import traceback # For detailed error logging
from fastapi.middleware.cors import CORSMiddleware # CORS用

app = FastAPI(title="REBA Evaluation API")

# -----------------------------
# モデル定義 (Pydantic)
# -----------------------------
class Landmark(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    visibility: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @validator('x', 'y', pre=True, always=True)
    def check_presence(cls, v, field):
        if v is None: raise ValueError(f"Field '{field.name}' must be present")
        return v

class CalibrationInputs(BaseModel):
    filmingSide: str = Field(..., description="left or right")
    neckRotation: float # Flag: >0 means rotated
    neckLateralBending: float # Flag: >0 means bent
    trunkLateralFlexion: float # Flag: >0 means bent
    loadForce: float # Represents score bracket start (0, 5, 11)
    shockForce: int = Field(..., description="Shock/Rapid force flag (0 or 1)") # ★ 衝撃力フラグ追加 ★
    postureCategory: str # "standingBoth", "standingOne", "sittingWalking"
    supportingLeg: Optional[str] = None # Supporting leg for standingOne
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

    @validator('supportingLeg')
    def supporting_leg_valid(cls, v):
        if v is not None and v not in ['left', 'right']: raise ValueError('Supporting leg must be "left", "right", or null')
        return v

    @validator('wristBaseScore')
    def wrist_base_score_valid(cls, v):
        if v not in [1, 2]: raise ValueError('Wrist Base Score must be 1 or 2')
        return v

    # ★ 衝撃力フラグのバリデータ追加 ★
    @validator('shockForce')
    def shock_force_valid(cls, v):
        if v not in [0, 1]: raise ValueError('Shock Force flag must be 0 or 1')
        return v

    # Consider adding validators for other flags (0/1) and floats if needed

class REBAInput(BaseModel):
    landmarks: List[Landmark]
    calibInputs: CalibrationInputs

# -----------------------------
# 2D Vector Calculation Helpers
# -----------------------------
def parse_point(p: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]: # ... (変更なし) ...
def vec_subtract_2d(p1_data: Dict, p2_data: Dict) -> Optional[Dict[str, float]]: # ... (変更なし) ...
def vec_dot_2d(v1: Dict, v2: Dict) -> float: # ... (変更なし) ...
def vec_cross_2d(v1: Dict, v2: Dict) -> float: # ... (変更なし) ...
def vec_magnitude_2d(v: Optional[Dict[str, float]]) -> float: # ... (変更なし) ...
def angle_between_2d_vectors(v1_data: Dict, v2_data: Dict) -> float: # ... (変更なし) ...
def angle_with_vertical(p1_data: Dict, p2_data: Dict, min_visibility: float = 0.5) -> float: # ... (変更なし) ...
def calculate_angle(p1_data: Dict, p2_data: Dict, p3_data: Dict, min_visibility: float = 0.5) -> float: # ... (変更なし) ...

# -----------------------------
# 角度計算 (Pure 2D)
# -----------------------------
def compute_all_angles(landmarks: List[Landmark], filming_side: str) -> Dict[str, Any]: # ... (変更なし) ...

# -----------------------------
# Revised Scoring Functions
# -----------------------------
def calc_neck_score_revised(angle_magnitude: float, is_extension: bool, rotationFlag: bool, sideBendFlag: bool) -> int: # ... (変更なし) ...
def calc_trunk_score_revised(angle_magnitude: float, is_extension: bool, rotationFlag: bool, sideBendFlag: bool) -> int: # ... (変更なし) ...
def calc_upper_arm_score_revised(angle_relative_to_trunk: float, is_extension: bool, upperArmCorrection: float, shoulderElevation: float, gravityAssist: float) -> int: # ... (変更なし) ...
def calc_leg_score_unified(postureCategory: str, kneeFlexAngle: Optional[float]) -> int: # ... (変更なし) ...
def calc_forearm_score(elbowAngle: Optional[float]) -> int: # ... (変更なし) ...
def calc_wrist_score(base_score_from_input: int, wristCorrectionFlag: float) -> int: # ... (変更なし) ...

# --- ★ 修正版: calc_load_score ★ ---
def calc_load_score(loadKgInput: float, shockForceFlag: int) -> int:
    """ Calculates Load/Force score including shock force correction """
    base_score = 0
    if loadKgInput < 5: base_score = 0      # < 5kg
    elif loadKgInput <= 10: base_score = 1 # 5-10kg (Input value 5 maps here)
    else: base_score = 2                   # > 10kg (Input value 11 maps here)

    # Add shock force correction (+1 if flag is 1)
    final_load_score = base_score + int(shockForceFlag)
    # Clamp the result to the possible range 0-3
    return max(0, min(3, final_load_score))

# -----------------------------
# Lookups & Helpers
# -----------------------------
tableA_Lookup = { ... } # (変更なし)
tableB_Lookup = { ... } # (変更なし)
tableC_Lookup = { ... } # (変更なし)

def lookup_score(table: Dict[str, int], key_parts: List[int], min_vals: List[int], max_vals: List[int]) -> int: # ... (変更なし) ...

# --- ★ 修正版: getScoreA ★ ---
def getScoreA(trunk: int, neck: int, leg: int, loadKgInput: float, shockForceFlag: int) -> int: # ★ shockForceFlag を追加
    """ Calculates Score A by looking up Table A and adding final load score """
    # Max component scores for lookup: Trunk=6, Neck=4, Leg=4
    table_a_score = lookup_score(tableA_Lookup, [trunk, neck, leg], [1, 1, 1], [6, 4, 4])
    # Calculate final load score including shock/rapid force
    final_load_score = calc_load_score(loadKgInput, shockForceFlag)
    # Add final load score to table score
    return table_a_score + final_load_score

def getScoreB(upperArm: int, forearm: int, wrist: int, coupling: int) -> int: # ... (変更なし) ...
def getTableCScore(scoreA: int, scoreB: int) -> int: # ... (変更なし) ...
def get_risk_level(score: int) -> str: # ... (変更なし) ...

# -----------------------------
# Final REBA Score Calculation Function
# -----------------------------
def get_final_reba_score(landmarks: List[Landmark], calib: CalibrationInputs) -> Dict[str, Any]:
    try:
        angles = compute_all_angles(landmarks, calib.filmingSide)
    except Exception as e: # Catch potential errors during angle calculation
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
        neckScore = calc_neck_score_revised(...) # (No change to call)
        trunkScore = calc_trunk_score_revised(...) # (No change to call)
        legScore = # ... (Calculate legScore using postureCategory, supportingLeg, knee angles - No change here) ...

        side_to_eval_arm = calib.filmingSide
        prefix = "L_" if side_to_eval_arm == "left" else "R_"
        upperArmScore = calc_upper_arm_score_revised(...) # (No change to call)
        forearmScore = calc_forearm_score(...) # (No change to call)
        wristScore = calc_wrist_score(calib.wristBaseScore, calib.wristCorrection) # (No change to call)

        # --- Combine Scores ---
        # ★★★ getScoreA に shockForce フラグを渡す ★★★
        scoreA = getScoreA(trunkScore, neckScore, legScore, calib.loadForce, calib.shockForce)

        scoreB = getScoreB(upperArmScore, forearmScore, wristScore, calib.coupling) # (No change)
        tableCScore = getTableCScore(scoreA, scoreB) # (No change)
        activityScore = calib.staticPosture + calib.repetitiveMovement + calib.unstableMovement # (No change)
        finalScore = tableCScore + activityScore # (No change)
        finalScore = max(1, min(15, finalScore)) # (No change)
        riskLevel = get_risk_level(finalScore) # (No change)

        response_data = { # (No structural change)
            "final_score": finalScore, "risk_level": riskLevel,
            "computed_angles": angles,
            "intermediate_scores": {
                 "neck": neckScore, "trunk": trunkScore, "leg": legScore,
                 "upperArm": upperArmScore, "forearm": forearmScore, "wrist": wristScore,
                 "scoreA": scoreA, "scoreB": scoreB, "tableC": tableCScore, "activity": activityScore,
                 # Optionally include final load score if needed for debugging:
                 # "load_score_final": calc_load_score(calib.loadForce, calib.shockForce)
            }
        }
        print("DEBUG: Returning data:", response_data)
        return response_data

    except Exception as e:
         print(f"ERROR: Unexpected error during REBA score calculation logic: {e}")
         traceback.print_exc()
         raise HTTPException(status_code=500, detail=f"Score calculation logic failed: {e}")

# -----------------------------
# API Endpoint (変更なし)
# -----------------------------
@app.post("/compute_reba")
async def compute_reba_endpoint(input_data: REBAInput): # ... (Error handling as before) ...

# -----------------------------
# CORS Middleware (変更なし)
# -----------------------------
origins = [ /* ... Your frontend URL ... */ ]
app.add_middleware( CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["POST", "GET"], allow_headers=["*"], )

# -----------------------------
# Root Endpoint (変更なし)
# -----------------------------
@app.get("/")
async def read_root(): return {"message": "REBA Evaluation API is running"}

# -----------------------------
# Optional: Uvicorn runner for local testing (変更なし)
# -----------------------------
# if __name__ == "__main__": # ...
