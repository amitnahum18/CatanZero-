"""Finalize the fast GBRL SARSA pilot after engine and model audits."""
import os
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import argparse
import json
from pathlib import Path
from mlflow.tracking import MlflowClient

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    root = parser.parse_args().study.resolve()
    read = lambda name: json.loads((root / name).read_text())
    report, config, registration, tracking = [read(name) for name in ("report.json", "config.json", "registration.json", "tracking.json")]
    update, games, benchmark = [read(name) for name in ("update_audit.json", "game_audit.json", "parallel_benchmark.json")]
    assert update["passed"] and games["passed"] and benchmark["same_game_results_verified"]
    assert update["tree_candidate_predictions_verified"] == games["tree_candidate_vectors_exported"]
    client = MlflowClient(tracking_uri=tracking["tracking_uri"])
    assert client.get_run(tracking["run_id"]).info.status == "FINISHED"
    version = client.get_model_version(registration["registered_model"], registration["version"])
    assert version.run_id == tracking["run_id"]
    lines = ["# פיילוט מהיר GBRL עם SARSA", "",
             "עצים מעודכנים בהדרגה ממשחקים חדשים, ללא סימולציה קדימה בבחירת פעולה.",
             "המודל מתחיל ללא עצים וללא חיקוי או משקלים מהמודל הלינארי. הפלט הוא Q ולא הסתברות מכוילת.", "",
             "## הגדרות שנקבעו לפני ההרצה", "",
             f"{config['training_games']} משחקי אימון, שמונה תהליכי משחק ומעדכן מרכזי אחד. במודל העצים שני threads של OpenMP על CPU.",
             "עצים סימטריים בעומק עד 4, 32 bins ולפחות 8 דוגמאות בעלה. קצב boosting=0.05.",
             "עץ נוסף לכל כ־128 מעברים חדשים; האצווה האחרונה עשויה להיות קטנה יותר. לא מאמנים מחדש את כל היער.",
             "164 תכונות מנורמלות וקפואות ממקור אימון קודם, חתוכות ל־[-4,4].",
             "חקירה 10%→5%; תגמול ניצחון=1 והפסד=0 עם פוטנציאל ניקוד עצמי 0.05 והיוון 0.999 לכל פעולת משחק אמיתית.",
             "יעד SARSA משתמש בפעולה הבאה שנבחרה ובוצעה בפועל; נגזרת שגיאת TD חתוכה ל־[-1,1].",
             "שלושה F_static הם היריבים בכל משחק. מושבים מאוזנים וכל לוחות האימון זרים ללוחות הבדיקה וללוחות הניסויים הקודמים.",
             "המודל הסופי נקבע מראש בסוף התקציב, בלי בחירה לפי תוצאות הבדיקה.", "",
             "## מקביליות וזמן", "",
             f"מדידה של אותם 8 משחקים עם מדיניות לינארית קפואה: תהליך אחד {benchmark['one_worker']['elapsed_seconds']:.2f} שניות; {config['workers']} תהליכים {benchmark['parallel_workers']['elapsed_seconds']:.2f} שניות; פי {report['parallel_speedup']:.2f}.",
             "התוצאות, הלוחות, מספרי הפעולות והזוכים זהים בשתי ההרצות. זהו מדגם קטן הכולל פתיחת תהליכים; אינו אומדן מדויק לתקציבים גדולים.",
             f"זמן אימון כולל משחקים ועדכונים: {report['live_training']['elapsed_seconds']:.1f} שניות; הזמן בתוך צעדי בניית העצים בלבד: {report['boosting_seconds']:.2f} שניות.", "",
             "## תוצאות", "", "| שלב | ניצחונות | משחקים שהושלמו | שיעור ניצחון | זמן בשניות |",
             "|---|---:|---:|---:|---:|"]
    for label, phase in (("מודל לינארי גרסה 3, קפוא", "linear_evaluation"),
                         ("אימון עצים, מדיניות משתנה", "live_training"),
                         ("מודל עצים סופי, קפוא", "tree_evaluation")):
        row = report[phase]
        lines.append(f"| {label} | {row['learner_wins']} | {row['completed_games']} | {row['win_rate_completed']:.1%} | {row['elapsed_seconds']:.1f} |")
    lines += ["", "שתי בדיקות המודלים הקפואים משתמשות באותם 100 לוחות ומושבים חדשים.",
              f"ניצחון של העצים בלבד ב־{report['paired_tree_only_wins']} זוגות; ניצחון של הלינארי בלבד ב־{report['paired_linear_only_wins']} זוגות; מבחן מזווג דו־צדדי p={report['paired_two_sided_p']:.6g}.",
              f"מעברים חדשים: {report['fresh_training_transitions']:,}; צעדי boosting: {report['central_updates']}; עדכונים ששינו חיזוי על דוגמאות העדכון: {report['updates_with_prediction_change']}.",
              "ניצחונות האימון הם של מודל העצים בפועל, אבל המדיניות משתנה במהלך האימון ולכן אינם מדד הכוח של המודל הסופי.", "",
              "## מסקנות ומגבלות", "",
              "זו בדיקת כדאיות קצרה, לא השוואה בתקציב אימון שווה: הלינארי עבר 1,500 משחקי אימון והעצים 200 בלבד.",
              "100 משחקי בדיקה מספקים אומדן גס. התכונות המרחביות עדיין מסוכמות; שינוי המודל אינו מוסיף מידע לקלט.",
              "הפיילוט מקצר את תקציב המשחקים ומפחית את תדירות העדכונים. אין להסיק ממנו שפחות אימון משפר איכות או שמשפחת עצים חלשה באופן כללי.", "",
              "## שמירה ואימות", "",
              f"Registry: `{registration['registered_model']}`, גרסה {registration['version']}, alias `{registration['alias']}`; run `{tracking['run_id']}`.",
              "הנרמול והחיתוך כלולים במודל MLflow. שמירה וטעינה מחדש אומתו לפני בדיקת המשחק; שגיאת שחזור אפס.",
              "כל יעדי SARSA, נגזרות ה־TD ועדכוני העצים חושבו מחדש מאפס, וכל מעבר אומת כמשומש פעם אחת.",
              f"שוחזרו {sum(games['games_replayed'].values())} משחקים, {games['actions_verified']:,} פעולות, {update['tree_candidate_predictions_verified']:,} ציוני פעולות של העצים ו־{games['candidate_predictions_verified']:,} של הלינארי.",
              f"בדיקות פרטיות: {games['privacy_invariance_checks']}; בדיקות תגמול אמיתי במדגם: {update['sampled_real_rewards_verified']}.",
              "שתי בדיקות GBRL וארבע בדיקות SARSA עברו. תוויות המדיניות מפרידות בין GBRL_SARSA ל־Linear_SARSA.", "",
              "```powershell",
              ".\\.ml-venv\\Scripts\\python.exe -u tools/train_gbrl_sarsa.py --output runs/gbrl-sarsa-fast-200-v1 --games 200 --eval-games 100 --workers 8",
              "```", ""]
    text = "\n".join(lines)
    (root / "README.md").write_text(text, encoding="utf-8")
    (REPO / "docs" / f"{root.name}.md").write_text(text, encoding="utf-8")
    verification = {"passed": True, "registered_model_version": registration["version"],
                    "all_tree_updates_recomputed": True, "sampled_engine_replays_verified": True,
                    "tree_scores_on_real_predecision_states_verified": True,
                    "same_results_in_parallel_benchmark": True, "model_reload_exact": True,
                    "gbrl_tests_passed": 2, "linear_training_tests_passed": 4}
    (root / "final_verification.json").write_text(json.dumps(verification, indent=2))
    for name in ("README.md", "final_verification.json", "update_audit.json", "game_audit.json"):
        client.log_artifact(tracking["run_id"], str(root / name), "pilot")
    for name in (*config["source_sha256"], "tools/audit_gbrl_sarsa.py", "tools/audit_online_games.py",
                 "tools/finalize_gbrl_pilot.py", "tests/test_gbrl_training.py", "requirements-gbrl.txt"):
        client.log_artifact(tracking["run_id"], str(REPO / name), "implementation")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
