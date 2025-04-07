# main.py (グラフ機能追加時点の状態)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Dict, Any
import math
from fastapi.middleware.cors import CORSMiddleware # CORS用

app = FastAPI(title="REBA Evaluation API")

# --- モデル定義 (変更なし) ---
class Landmark(BaseModel): # ...
class CalibrationInputs(BaseModel): # ... (couplingのvalidatorは0-3を許可)
    # ... fields ...
    @validator('coupling')
    def coupling_must_be_valid(cls, v):
        if not 0 <= v <= 3: raise ValueError('Coupling score must be between 0 and 3')
        return v
    # ... other validators ...
class REBAInput(BaseModel): # ...

# --- 角度計算ユーティリティ (変更なし) ---
def calculate_angle(p1: Dict, p2: Dict, p3: Dict, min_visibility: float = 0.5) -> float: # ...
def angle_with_vertical(p1: Dict, p2: Dict, min_visibility: float = 0.5) -> float: # ...

# --- ランドマークから角度計算 (変更なし) ---
def compute_all_angles(landmarks: List[Landmark]) -> Dict[str, float]: # ...

# --- REBAスコア計算関数群 (変更なし) ---
def calc_neck_score(neckAngle: float, rotationFlag: bool, sideBendFlag: bool) -> int: # ...
def calc_trunk_score(trunkFlexAngle: float, rotationFlag: bool, sideBendFlag: bool) -> int: # ...
def calc_leg_score_unified(postureCategory: str, kneeFlexAngle: float) -> int: # ...
def calc_upper_arm_score(upperArmAngle: float, upperArmCorrection: float, shoulderElevation: float, gravityAssist: float) -> int: # ...
def calc_forearm_score(elbowAngle: float) -> int: # ...
def calc_wrist_score(wristAngle: float, wristCorrection: float) -> int: # ...
def calc_load_score(loadKg: float) -> int: # ...

# --- ルックアップテーブル & ヘルパー (変更なし) ---
tableA_Lookup = { ... }
tableB_Lookup = { ... }
tableC_Lookup = { ... }
def lookup_score(table: Dict, key_parts: List[int], min_vals: List[int], max_vals: List[int]) -> int: # ...
def getScoreA(trunk: int, neck: int, leg: int, loadKg: float) -> int: # ...
def getScoreB(upperArm: int, forearm: int, wrist: int, coupling: int) -> int: # ...
def getTableCScore(scoreA: int, scoreB: int) -> int: # ...
def get_risk_level(score: int) -> str: # ...

# --- 最終REBAスコア算出関数 (変更なし、intermediate_scoresを返す) ---
def get_final_reba_score(landmarks: List[Landmark], calib: CalibrationInputs) -> Dict[str, Any]:
    # ... (計算処理) ...
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
    print("DEBUG: Returning data:", response_data) # デバッグ用ログ
    return response_data

# --- API エンドポイント (変更なし) ---
@app.post("/compute_reba")
async def compute_reba_endpoint(input_data: REBAInput):
    try:
        result = get_final_reba_score(input_data.landmarks, input_data.calibInputs)
        return result
    except HTTPException as e: raise e
    except ValidationError as e: raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e: # ... (エラーハンドリング) ...

# --- CORS ミドルウェア設定 (フロントエンドURL設定済み) ---
origins = [
    "http://localhost", "http://127.0.0.1",
    # ★★★ 正しいフロントエンドURLに設定済みであること ★★★
    "https://reba-1.onrender.com",
]
app.add_middleware( CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["POST", "GET"], allow_headers=["*"], )

# --- ルートエンドポイント (変更なし) ---
@app.get("/")
async def read_root(): return {"message": "REBA Evaluation API is running"}

