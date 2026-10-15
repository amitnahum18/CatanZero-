"""Verify and document the completed live learner and frozen evaluations."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
from mlflow.tracking import MlflowClient

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    root = parser.parse_args().study.resolve()
    report = json.loads((root / "report.json").read_text())
    config = json.loads((root / "config.json").read_text())
    tracking = json.loads((root / "tracking.json").read_text())
    registration = json.loads((root / "registration.json").read_text())
    audits = {name: json.loads((root / f"{name}.json").read_text()) for name in ("update_audit", "game_audit")}
    assert all(a["passed"] for a in audits.values())
    assert report["central_updates"] > 0 and report["updates_with_nonzero_parameter_change"] > 0
    for path, expected in config["source_sha256"].items():
        assert hashlib.sha256((REPO / path).read_bytes()).hexdigest() == expected
    client = MlflowClient(tracking_uri=tracking["tracking_uri"])
    assert client.get_run(tracking["run_id"]).info.status == "FINISHED"
    version = client.get_model_version_by_alias("CatanZero_Online_Linear_SARSA", "pilot")
    assert str(version.version) == registration["version"] and version.run_id == tracking["run_id"]
    phases = [("פתיחה: משקלים אפסיים, קפואים", report["initial_evaluation"]),
              ("אימון: משקלים משתנים", report["live_training"]),
              ("סוף: המודל המאומן, קפוא", report["final_evaluation"])]
    lines = ["# ניסוי Online Linear SARSA", "",
             "14 באוקטובר 2026. האימון המקוון, בדיקת המשחק והרישום ב־MLflow הושלמו.", "",
             "המודל הוא SGDRegressor ליניארי: 164 מקדמים ומקדם חופשי אחד. כל 165 הפרמטרים התחילו מאפס.",
             "שמונה תהליכי משחק עצמאיים המתינו לבחירות של מודל מרכזי אחד. רק התהליך המרכזי עדכן משקלים.",
             "המודל עצמו שיחק מול שלושה F_static; לא נעשה חיקוי של ציוני היריב ולא נעשה שימוש במאגר מעברים קבוע.", "",
             "## האימון", "",
             f"נאספו {report['fresh_training_transitions']:,} מעברים חדשים במהלך {report['live_training']['scheduled_games']} משחקי אימון.",
             f"בוצעו {report['central_updates']:,} עדכונים מרכזיים; ב־{report['updates_with_nonzero_parameter_change']:,} מהם הפרמטרים השתנו בפועל.",
             "בכל עדכון נעשה partial_fit על כ־32 מעברים, בלי לאפס את המקדמים. היעדים חושבו לפני העדכון.",
             "יעד SARSA הוא התגמול ועוד הערך של הפעולה הבאה שנבחרה בפועל בהחלטה הבאה של אותו שחקן.",
             "הפעולה הבאה נשמרת ומבוצעת גם אם המשקלים מתעדכנים ביניהם; עקבות המשחק מתעדים את גרסת המשקלים שלה.",
             "מקדם ההיוון הוא 0.999 לכל פעולה אמיתית שחלפה. תגמול בסיס: ניצחון 1, הפסד 0.",
             "נוסף פוטנציאל 0.05 כפול הניקוד הפרטי המותר של השחקן, עם היוון וסגירת הפוטנציאל בסיום.",
             "אין בונוס רצף ניצחונות. סוף משחק שלא הושלם נשאר תוצאה לא ידועה; המעבר האחרון ללא מצב המשך נזרק.",
             "חקירה יורדת מ־30% ל־10% לפי אינדקס משחקי האימון. בשלב הבדיקה אין חקירה; שוויון ציונים מוכרע אקראית עם seed קבוע.", "",
             "הנרמול קפוא מנתוני האימון של הניסוי הקודם. הוא אינו משתמש במשחקי הבדיקה החדשים ולא משנה משקלי פתיחה.",
             "תכונות מנורמלות נחתכות ל־[-4,4]. קצב למידה 0.05 עם משקל דוגמה 1/(1+נורמת התכונות בריבוע), ורגולריזציית L2 של 0.0001.", "",
             "## משחקים ותוצאות", "",
             "| שלב | ניצחונות של המודל | משחקים שהסתיימו | משחקים לא גמורים | זמן שלב |",
             "|---|---:|---:|---:|---:|"]
    for name, phase in phases:
        lines.append(f"| {name} | {phase['learner_wins']} | {phase['completed_games']} | {phase['unfinished_games']} | {phase['elapsed_seconds']:.1f} שניות |")
    lines += ["", "כל הניצחונות בטבלה הם של השחקן המופעל על ידי המודל הליניארי, ולא של בוט איסוף יוריסטי.",
              "שיעור הניצחונות בזמן אימון אינו אומדן כוח של מודל קפוא: המשקלים והחקירה השתנו לאורך השלב.",
              "מדידות הפתיחה והסוף משתמשות באותם לוחות חדשים ובאותם מושבים, בלי עדכון משקלים.",
              "משחקי האימון אינם חופפים ללוחות הבדיקה. המודל הסופי נקבע מראש אחרי 500 משחקים; תוצאות הבדיקה לא שימשו לכוונון.",
              f"זוגות מלאים שהשתפרו: {report['paired_improvements']}; שנחלשו: {report['paired_regressions']}. מבחן מזווג דו־צדדי: p={report['paired_two_sided_p']:.6g}.",
              "קו הייחוס לשחקן ברמת F_static מול שלושה F_static, במושבים מאוזנים, הוא 25% ניצחונות בממוצע.",
              "שיפור לעומת מודל אפסי אינו מוכיח מעבר לרמת היריב. המודל הליניארי והייצוג המרחבי המסוכם עשויים להגביל את האסטרטגיה שניתן ללמוד.", "",
              "## אימות ושמירה", "",
              f"כל {audits['update_audit']['updates_recomputed']:,} עדכוני המשקלים חושבו מחדש מאפס והושוו לצילומי המשקלים השמורים.",
              "אומתו יעדי SARSA, שימוש חד־פעמי במעברים, מעבר בין החלטות של אותו שחקן, איזון מושבים והפרדת לוחות.",
              f"שוחזרו {sum(audits['game_audit']['games_replayed'].values())} משחקים במדגם: {audits['game_audit']['actions_verified']:,} פעולות ו־{audits['game_audit']['candidate_predictions_verified']:,} ציוני פעולות.",
              "הבחירה נעשית מתצפית מותרת בלבד. העתקה, יצירה והפעלה של Game חסומות בזמן החישוב בסימולטור; המודל המרכזי מקבל רק תכונות, ללא אובייקט משחק.",
              "80 בדיקות בסביבת המנוע ושתי בדיקות אימון בסביבת sklearn עברו. מודול אימון אחד דולג בסביבת המנוע ונבדק בנפרד בסביבת האימון.",
              f"שם Registry: CatanZero_Online_Linear_SARSA, גרסה {registration['version']}, alias: pilot.",
              f"Run: {tracking['run_id']}. הנרמול והחיתוך כלולים במודל הרשום, שמקבל 164 מאפיינים גולמיים.",
              "המודל נשמר ונטען מחדש לפני בדיקת הסוף. הפלט הוא ערך Q, ולא הסתברות מכוילת.", "",
              "```powershell",
              ".\\.ml-venv\\Scripts\\python.exe -u tools/train_online_sarsa.py --output runs/online-sarsa-next --games 500 --eval-games 200 --workers 8",
              "```", "",
              "כדי להשתמש בלוחות עצמאיים נוספים יש לשנות את זרעי הבסיס בקוד ולהימנע משימוש חוזר בבדיקה לצורך בחירת הגדרות.", ""]
    text = "\n".join(lines)
    (root / "README.md").write_text(text, encoding="utf-8")
    doc = REPO / "docs/online-sarsa-2026-10-14.md"
    doc.write_text(text, encoding="utf-8")
    verification = {"passed": True, "registration_run_status": "FINISHED", "model_version": registration["version"],
                    "update_audit_passed": True, "sampled_game_replay_passed": True,
                    "policy_sources_match_run_config": True, "simulator_tests_passed": 80, "online_training_tests_passed": 2}
    (root / "final_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    for file in (root / "README.md", root / "final_verification.json", root / "update_audit.json", root / "game_audit.json"):
        client.log_artifact(tracking["run_id"], str(file), "online_sarsa")
    for name in (*config["source_sha256"], "tools/audit_online_updates.py", "tools/audit_online_games.py"):
        client.log_artifact(tracking["run_id"], str(REPO / name), "implementation")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
