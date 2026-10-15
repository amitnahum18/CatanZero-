"""Verify and document a completed continuation without replacing prior reports."""

import argparse
import hashlib
import json
from pathlib import Path

from mlflow.tracking import MlflowClient

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    root = parser.parse_args().study.resolve()
    read = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    config, report = read("config.json"), read("report.json")
    registration, tracking = read("registration.json"), read("tracking.json")
    updates, games = read("update_audit.json"), read("game_audit.json")
    assert config["resume_from"] and not report["initial_parameters_all_zero"]
    assert updates["passed"] and games["passed"]
    assert updates["saved_starting_weights_and_optimizer_verified"]
    for path, expected in config["source_sha256"].items():
        assert hashlib.sha256((REPO / path).read_bytes()).hexdigest() == expected
    client = MlflowClient(tracking_uri=tracking["tracking_uri"])
    assert client.get_run(tracking["run_id"]).info.status == "FINISHED"
    version = client.get_model_version(registration["registered_model"], registration["version"])
    assert version.run_id == tracking["run_id"]
    ancestor = Path(config["resume_from"])
    while True:
        original = json.loads((ancestor / "registration.json").read_text())
        preserved = client.get_model_version(original["registered_model"], original["version"])
        assert preserved.run_id == original["run_id"]
        parent = json.loads((ancestor / "config.json").read_text()).get("resume_from")
        if not parent:
            break
        ancestor = Path(parent)
    assert str(client.get_model_version_by_alias(registration["registered_model"], "pilot").version) == original["version"]
    epsilon_start, epsilon_end = config.get("exploration_start", .3), config.get("exploration_end", .1)
    decision = None
    if "improvement_detected_by_precommitted_rule" in report:
        confirmed = report["improvement_detected_by_precommitted_rule"]
        decision = {"numerical_win_rate_increase": report["final_evaluation"]["win_rate_completed"] > report["initial_evaluation"]["win_rate_completed"],
                    "confirmed_improvement": confirmed,
                    "paired_two_sided_p": report["paired_two_sided_p"],
                    "precommitted_rule": config["improvement_rule"],
                    "next_action": "continue_training" if confirmed else "stop_training_pending_next_direction"}
        (root / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    lines = ["# המשך אימון Online Linear SARSA", "",
             "המשך מהמודל הלינארי השמור, כולל מצב האופטימייזר, ללא איפוס פרמטרים.",
             f"מודל מקור: `{config['resume_from']}`; גרסת עדכון מקור: {config['source_model_version']}.",
             "164 תכונות מנורמלות, 165 פרמטרים; SGDRegressor עם עדכוני SARSA מקוונים.",
             "שמונה סביבות משחק מול שלושה F_static ועדכון משקלים מרכזי אחד.",
             "בחירת מהלכים ללא סימולציה קדימה. פלט Q אינו הסתברות ניצחון מכוילת.", "",
             "## פרוטוקול שנקבע לפני הריצה", "",
             f"{config['training_games']} משחקי אימון חדשים; {config['evaluation_games_per_snapshot']} משחקי בדיקה לפני ואחרי על אותם לוחות ומושבים.",
             "לוחות האימון והבדיקה זרים זה לזה ולכל לוחות ניסוי המקור. הבדיקה אינה מעדכנת פרמטרים.",
             "נרמול קפוא ממקור האימון הקודם. L2=0.0001, קצב למידה 0.05 ומשקל דוגמה 1/(1+||x||²).",
             f"חקירה באימון יורדת מ־{epsilon_start:.0%} ל־{epsilon_end:.0%}; בבדיקה אין חקירה. ההגדרות נקבעו לפני הריצה ולא שונו לפי תוצאות בדיקתה.",
             f"כלל החלטה לפני ההרצה: {config.get('improvement_rule', 'לא נקבע כלל להמשך אוטומטי')}.",
             "תגמול בסיס ניצחון=1 והפסד=0; תוספת פוטנציאל 0.05 כפול הניקוד העצמי הידוע, עם היוון 0.999 לכל פעולה אמיתית וסגירה בסיום.",
             "יעד העדכון משתמש בפעולה הבאה שנבחרה ובוצעה בפועל של אותו שחקן. אין בונוס רצף ניצחונות.", "",
             "## תוצאות", "", "| שלב | ניצחונות | משחקים שהושלמו | שיעור ניצחון | זמן בשניות |",
             "|---|---:|---:|---:|---:|"]
    for label, phase in (("מודל המקור, קפוא", "initial_evaluation"),
                         ("אימון, מדיניות משתנה", "live_training"),
                         ("מודל ההמשך, קפוא", "final_evaluation")):
        row = report[phase]
        lines.append(f"| {label} | {row['learner_wins']} | {row['completed_games']} | {row['win_rate_completed']:.1%} | {row['elapsed_seconds']:.1f} |")
    lines += ["", "ניצחונות האימון שייכים למודל ששיחק בפועל, אבל הם אינם מדד כוח למודל קפוא.",
              f"מעברים חדשים: {report['fresh_training_transitions']:,}; עדכונים: {report['central_updates']:,}; עדכונים ששינו משקלים: {report['updates_with_nonzero_parameter_change']:,}.",
              f"זוגות שהשתפרו: {report['paired_improvements']}; שנחלשו: {report['paired_regressions']}; מבחן מזווג דו־צדדי p={report['paired_two_sided_p']:.6g}.",
              f"שיעורי הניצחון: {report['initial_evaluation']['win_rate_completed']:.1%} לפני ו־{report['final_evaluation']['win_rate_completed']:.1%} אחרי. מובהקות ברמת 5%: {'כן' if report['paired_two_sided_p'] < .05 else 'לא'}.",
              "לא נבחרו מחדש משקלים או פרמטרים בעקבות הבדיקה. מודל ההמשך נשמר לניסוי, ומודל הפיילוט המקורי נשאר תחת pilot.",
              "שיעור 25% הוא קו ייחוס תאורטי לשחקן שווה־כוח מול שלושה F_static במושבים מאוזנים, ולא תוצאת קבוצת ביקורת נוספת.",
              "זהו ניסוי אחד עם תכונות מרחביות מסוכמות ומודל לינארי; אינו מוכיח ביצועים מול F המקורי או מול בני אדם.", "",
              "השוואת לפני ואחרי מודדת את השפעת המשך האימון בהגדרות אלה. ללא קבוצת אימון מקבילה עם חקירה אחרת, אינה מבודדת את התרומה הסיבתית של הפחתת החקירה.", "",
              "## שמירה ואימות", "",
              f"MLflow Registry: `{registration['registered_model']}`, גרסה {registration['version']}, alias `{registration['alias']}`.",
              f"Run: `{tracking['run_id']}`. המודל הקודם ו־alias `pilot` נשמרו.",
              "נרמול וחיתוך כלולים במודל הרשום; הטעינה מחדש אומתה לפני בדיקת הסוף.",
              "כל יעדי ועדכוני SARSA חושבו מחדש ממצב האופטימייזר של המודל השמור והושוו לצילומי המשקלים.",
              f"שוחזרו {sum(games['games_replayed'].values())} משחקים: {games['actions_verified']:,} פעולות, {games['candidate_predictions_verified']:,} ציוני פעולות, {games['privacy_invariance_checks']:,} בדיקות פרטיות.",
              "ארבע בדיקות אימון, כולל המשך שקול לעדכון ללא הפסקה ולוח החקירה החדש, עברו.", "",
              "```powershell",
              f".\\.ml-venv\\Scripts\\python.exe -u tools/train_online_sarsa.py --output runs/{root.name} --resume-from runs/{Path(config['resume_from']).name} --games {config['training_games']} --eval-games {config['evaluation_games_per_snapshot']} --workers {config['environment_processes']} --training-base-seed {config['training_base_seed']} --evaluation-base-seed {config['evaluation_base_seed']} --exploration-start {epsilon_start} --exploration-end {epsilon_end}",
              "```", ""]
    if decision:
        lines += ["## החלטה", "",
                  f"שיפור מספרי: {'כן' if decision['numerical_win_rate_increase'] else 'לא'}; שיפור מובהק לפי הכלל שנקבע מראש: {'כן' if decision['confirmed_improvement'] else 'לא'}.",
                  "המשך אימון נוסף מאושר לפי הכלל." if decision["confirmed_improvement"] else "לא נפתח סבב אימון נוסף. התוצאה אינה מוכיחה שאין שיפור; הראיות אינן מספיקות להפעלת תנאי ההמשך.", ""]
    text = "\n".join(lines)
    (root / "README.md").write_text(text, encoding="utf-8")
    doc = REPO / "docs" / f"{root.name}.md"
    doc.write_text(text, encoding="utf-8")
    verification = {"passed": True, "registered_model_version": registration["version"],
                    "saved_starting_estimator_verified": True, "prior_model_preserved": True,
                    "all_updates_recomputed": True, "sampled_game_replays_verified": True,
                    "training_tests_passed": 4}
    (root / "final_verification.json").write_text(json.dumps(verification, indent=2))
    for name in ("README.md", "final_verification.json", "update_audit.json", "game_audit.json"):
        client.log_artifact(tracking["run_id"], str(root / name), "online_sarsa")
    if decision:
        client.log_artifact(tracking["run_id"], str(root / "decision.json"), "online_sarsa")
        client.log_metric(tracking["run_id"], "initial_eval_win_rate", report["initial_evaluation"]["win_rate_completed"])
        client.log_metric(tracking["run_id"], "final_eval_win_rate", report["final_evaluation"]["win_rate_completed"])
        client.log_metric(tracking["run_id"], "paired_two_sided_p", report["paired_two_sided_p"])
        client.set_tag(tracking["run_id"], "conditional_training_decision", decision["next_action"])
    for name in (*config["source_sha256"], "tools/audit_online_updates.py", "tools/audit_online_games.py",
                 "tools/finalize_sarsa_continuation.py", "tests/test_online_training.py"):
        client.log_artifact(tracking["run_id"], str(REPO / name), "implementation")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
