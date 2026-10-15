"""Save completed training only after the user's deadline stop; no evaluation."""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
from mlflow.tracking import MlflowClient
import numpy as np

root = Path(__file__).resolve().parents[1] / "runs/fqi-pilot-500-v1"
tracking = json.loads((root / "tracking.json").read_text())
collection = json.loads((root / "collection/summary.json").read_text())
history = json.loads((root / "training_history.json").read_text())
arrays = np.load(root / "transitions.npz")
report = {"status": "stopped_by_user_after_training_before_game_evaluation",
          "algorithm": "offline Fitted Q Iteration with ExtraTrees", "training_iterations": len(history),
          "collection": collection, "training_transitions": int(arrays["train_mask"].sum()),
          "validation_transitions": int((~arrays["train_mask"]).sum()),
          "test_games_run": 0, "beats_F_static": None,
          "mlflow_run_id": tracking["run_id"], "model_artifacts_logged": True,
          "sklearn_registry_model_logged": False,
          "original_logging_error": "MLflow default skops format rejected sklearn.tree._tree.Tree; native and portable models saved as artifacts instead",
          "model_sha256": hashlib.sha256((root / "model.joblib").read_bytes()).hexdigest()}
(root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
doc = """# פיילוט ExtraTrees עם FQI

לפי הוראת המשתמש לעצור בדדליין, ההרצה הסתיימה לאחר איסוף הנתונים והאימון, לפני בדיקת משחק.
זהו Offline RL: הנתונים נאספו במשחקי F_static עם 10% חקירה אצל השחקן הנלמד, ולא במשחקי המודל החדש.

נאספו 500 משחקים שהסתיימו, ב־500 לוחות ייחודיים ובמושבים מאוזנים. האיסוף לקח 185.26 שניות.
נשמרו 217,797 פעולות ו־52,149 החלטות של השחקן הנלמד. 400 משחקים שימשו לאימון ו־100 לוולידציה.
המודל התחיל מחיקוי דירוג היוריסטיקה, ולאחר מכן ביצע שישה סבבי Fitted Q Iteration.
כל מעבר מקשר להחלטה הבאה של אותו שחקן, או לסוף המשחק; תגמול סופי 1 לניצחון ו־0 להפסד,
עם עיצוב תגמול המבוסס על פוטנציאל נקודות. חישוב המשך הערך נעשה באימון בלבד, מתוך הנתונים.
זמן הבחירה חסום בפני Game.copy, Game.execute ויצירת Game. מנוע המשחק מבצע רק את המהלך האמיתי שנבחר.

מודל ExtraTrees כולל 48 עצים, עומק מרבי 8 ומינימום 12 דוגמאות בעלה. הנרמול התבסס על האימון בלבד.
בדיקת יצוא הפלט לפורמט נייד עברה לפני ניסיון השמירה ב־MLflow, ושתי בדיקות חדשות של האיסוף עברו.
ביקורת כל 500 הקבצים בדקה מזהים, החלטות של אותו שחקן ומושבים; שוחזרו 4,741 פעולות במדגם משחקים.

model.joblib הוא המודל המקורי; model.json הוא פורמט חיזוי ללא sklearn; normalizer.json מגדיר את הנרמול.
ניסיון רישום המודל באמצעות ברירת המחדל skops של MLflow נכשל. הקבצים נשמרו כארטיפקטים של ה־run
המקורי, שמסומן FAILED עקב שגיאת הרישום, עם תג המבהיר שהאימון הסתיים והבדיקה לא התחילה.
לא נוצרה גרסת Registry ולא שונה alias של מודל קודם.

אין תוצאות של 200 משחקי בדיקה ואין ראיות שהמודל מנצח את F_static.
124 הניצחונות מתוך 500 הם תוצאות מדיניות האיסוף היוריסטית עם חקירה, ולא ניצחונות המודל שאומן.
המשך אפשרי הוא תיקון רישום המודל שכבר נשמר והערכתו על 200 לוחות חדשים. הוא לא בוצע בעקבות העצירה.
"""
(root / "README.md").write_text(doc, encoding="utf-8")
client = MlflowClient(tracking_uri=tracking["tracking_uri"])
for file in root.iterdir():
    if file.is_file():
        client.log_artifact(tracking["run_id"], str(file), "completed_training")
client.log_artifact(tracking["run_id"], str(root / "collection/summary.json"), "completed_training/collection")
client.set_tag(tracking["run_id"], "pilot_status", report["status"])
client.set_tag(tracking["run_id"], "model_evaluation_games", "0")
client.set_tag(tracking["run_id"], "completed_fqi_iterations", str(len(history)))
client.log_metric(tracking["run_id"], "training_transitions", report["training_transitions"])
print(json.dumps(report, indent=2), flush=True)
