// script.js 内の getCalibrationInputs 関数を以下のように修正

function getCalibrationInputs() {
  const names = [
    "filmingSide", "neckRotation", "neckLateralBending", "trunkLateralFlexion",
    "loadForce", "postureCategory", "upperArmCorrection", "shoulderElevation",
    "gravityAssist", "wristCorrection", "staticPosture", "repetitiveMovement",
    "unstableMovement", "coupling"
  ];
  const data = {};
  let errorOccurred = false;

  console.log("--- Checking Calibration Inputs ---"); // 確認開始ログ

  for (const name of names) {
    const element = document.querySelector(`input[name="${name}"]:checked`);
    if (element) {
      // filmingSide と postureCategory 以外は数値に変換
      if (name === "filmingSide" || name === "postureCategory") {
        data[name] = element.value;
      } else {
        data[name] = Number(element.value);
      }
      // console.log(`Input found for name="${name}": ${data[name]}`); // 個別成功ログ (任意)
    } else {
      // ★ 要素が見つからない場合、エラーログを出力 ★
      console.error(`ERROR in getCalibrationInputs: Could not find checked input for name="${name}"`);
      errorOccurred = true;
      // ここで処理を中断せず、他の項目もチェックし続ける
    }
  }

  console.log("--- Finished Checking Inputs ---"); // 確認終了ログ

  // エラーが発生していたら undefined を返す (元の動作に合わせる)
  if (errorOccurred) {
    return undefined;
  }

  console.log("Successfully obtained calibration inputs:", data); // 全て成功した場合のログ
  return data; // 全て成功した場合のみデータを返す
}
