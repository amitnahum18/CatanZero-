# CatanZero

## Play online

[Play CatanZero](https://amitnahum18.github.io/CatanZero-/) — one human against
1–3 Model 402 bots. The game runs entirely in your browser, with local autosave.
The first visit downloads the game engine; no server account is needed.

[Deployment and build instructions](docs/github-pages.md) ·
[Repository layout](docs/repository-layout.md) ·
[Selected model](models/champion402/README.md)

## משחק מול הבוטים

להפעלת אתר המשחק המקומי, עם בחירת מספר יריבים:

```powershell
.\play.cmd
```

פותחים את `http://127.0.0.1:8787/`, בוחרים 1–3 בוטים ומשחקים על הלוח.
כל הבוטים משתמשים במודל 402, ללא בחירת רמת קושי. המשחק נשמר אוטומטית.
[הוראות משחק והפעלה](docs/human-play.md).

תיוג נוסף באמצעות השוואת חלופות והמשכים עד ניצחון זמין ב־
[docs/search-labels.md](docs/search-labels.md). הוא שומר את הציונים המקוריים,
מוסיף אומדני ניצחון ואי־ודאות, ומריץ את ההמשכים מקומית במקביל.

חישוב תיוג משוקלל של יוריסטיקה ותוצאה סופית זמין במודול `catanzero.scoring`.
ראו [הגדרה, קלט ודוגמאות הרצה](docs/weighted-scoring.md).

איסוף משחקים ודוגמאות עם יוריסטיקה ושקלול זמין במודול `catanzero.dataset`.
ראו [הרצת פיילוט, מבנה הדאטה ומגבלות](docs/dataset-pilot.md).
אפשר להוסיף `--workers 4` או `--workers 8` לפקודת `catanzero.dataset`
כדי לאסוף ולתייג משחקים במקביל על המחשב המקומי.

סביבת סימולציה של קאטן, מבוססת על [Catanatron](https://github.com/bcollazo/catanatron).
המודל המומלץ כרגע הוא 402: שילוב של LightGBM ו־ExtraTrees, שניצח 56.4% מול F אחד
ב־500 משחקים דו־שחקניים ו־31.6% מול שלושה F ב־500 משחקי ארבעה שחקנים.
לסיכום ההישגים, הניסויים שנכשלו והמודל הנבחר, ראו [מצב הפרויקט](docs/PROJECT_STATUS.md).

## הרצה מהירה ב־Windows

מתוך תיקיית הפרויקט ב־PowerShell:

לצפייה במשחק האחרון שנשמר:

```powershell
.\watch.cmd
```

הפקודה מפעילה שרת מקומי ופותחת בדפדפן את `http://127.0.0.1:8765/`.
משאירים את חלון ההרצה פתוח בזמן הצפייה. `Ctrl+C` עוצר את השרת.
השרת נגיש רק מהמחשב המקומי ומגיש רק את תצוגת המשחק.

לצפייה במשחק מסוים:

```powershell
.\watch.cmd runs\my-run\game_0000.json
```

להרצת משחק חדש:

```powershell
.\run.cmd --seed 42
```

הסביבה `.venv` כבר הותקנה במחשב הנוכחי. הסקריפט משתמש בה ישירות, ללא צורך
בהפעלת `Activate.ps1` או בשינוי מדיניות הרצת סקריפטים.

משחק מלא עם שמירת מצב הלוח וכל המהלכים:

```powershell
.\run.cmd --seed 42 --save-games
```

משחק מלא עם תצוגת לוח שאפשר לנגן בדפדפן:

```powershell
.\run.cmd --seed 42 --replay
```

פותחים את `game_0000.html` בתיקיית התוצאות. התצוגה כוללת את הלוח, הכבישים,
היישובים, הערים, השודד, המשאבים והניקוד בכל מהלך. אפשר לנגן ולעצור, לבחור מהירות,
לגרור את ציר הזמן או להתקדם מהלך אחר מהלך. הקובץ עצמאי ועובד ללא אינטרנט או שרת.

ליצירת תצוגה ממשחק שכבר נשמר:

```powershell
.\.venv\Scripts\python.exe -m catanzero.viewer runs\my-run\game_0000.json
```

התצוגה משחזרת את תוצאות הקוביות והקלפים שהוקלטו ובודקת שמצב הסיום זהה למשחק השמור.

סדרת 100 משחקים:

```powershell
.\run.cmd --games 100 --seed 42
```

הדפסת כל פעולה, כולל תוצאות הקוביות ופעולות אקראיות אחרות:

```powershell
.\run.cmd --seed 42 --trace
```

הפקודה המקבילה ללא הסקריפט:

```powershell
.\.venv\Scripts\python.exe -m catanzero --seed 42
```

הפקודות `run.cmd` ו־`watch.cmd` מיועדות לסימולציה אוטומטית ולשידור חוזר.
למשחק אנושי מול הבוטים משתמשים ב־`play.cmd` כפי שמוסבר בתחילת הדף.

## התקנה מחדש

נדרשים Python 3.11 ומעלה, Git ו־[uv](https://docs.astral.sh/uv/).
מתוך תיקיית הפרויקט:

```powershell
uv sync --locked --cache-dir .uv-cache
.\run.cmd --seed 42
```

התקנה ראשונה זקוקה לחיבור אינטרנט. אחרי ההתקנה אפשר להריץ משחקים ללא אינטרנט.
התלויות נעולות ב־`uv.lock`, והמנוע נעול ל־commit
`ecf931181b9a65bb4116a2153fb78c16f1438e00` של Catanatron 3.3.0.
החבילה העדכנית מותקנת מהמאגר הרשמי; משתמשים רק בתלויות הליבה.

## הגדרות ותוצאות

ברירת המחדל היא לוח הבסיס, ארבעה שחקנים אקראיים, סדר ישיבה מעורבב ויעד של 10
נקודות ניצחון. המנוע מטפל בהקמה, קוביות, משאבים, בנייה, שודד, קלפי פיתוח,
הדרך הארוכה ביותר, הצבא הגדול ביותר ומסחר עם הבנק והנמלים.
השחקנים האקראיים הנוכחיים אינם יוזמים הצעות מסחר בין שחקנים.

| אפשרות | ברירת מחדל | משמעות |
| --- | --- | --- |
| `--games N` | 1 | מספר משחקים ברצף |
| `--players N` | 4 | 2, 3 או 4 שחקנים; 2 הוא וריאנט סימולציה על לוח הבסיס |
| `--seed N` | נוצר אוטומטית | seed למשחק הראשון; בכל משחק נוסף מוסיפים 1 |
| `--max-turns N` | 1000 | מגבלת תורות שהסתיימו, לפי ספירת המנוע |
| `--max-actions N` | 100000 | מגבלת החלטות/פעולות, כולל הקמה ופעולות מחוץ לתור |
| `--output PATH` | תיקייה חדשה תחת `runs/` | חייב להיות נתיב שעדיין אינו קיים |
| `--save-games` | כבוי | שמירת מצב מנוע מלא ויומן פעולות לכל משחק |
| `--replay` | כבוי | שמירת משחקים ותצוגות לוח עצמאיות ב־HTML |
| `--trace` | כבוי | הדפסת כל פעולה לטרמינל |

בכל הרצה נוצרות התוצאות הבאות:

- `config.json`: הגדרות, seed, גרסת Python וגרסת המנוע.
- `results.jsonl`: שורה לכל משחק, עם מנצח, ניקוד, סדר ישיבה, מספר תורות ופעולות וזמן ריצה.
- `summary.json`: מספר ניצחונות לפי צבע, משחקים שהושלמו ומשחקים שנעצרו במגבלה.
- `game_0000.json` וכו׳: עם `--save-games`, מצב הלוח, הקלפים, מצב המחולל האקראי והיסטוריית הפעולות בפורמט של המנוע.
- `game_0000.html` וכו׳: עם `--replay`, תצוגת משחק אינטראקטיבית בדפדפן.

קובצי מצב מלאים כוללים גם מידע מוסתר של השחקנים ומיועדים לבדיקה ולניתוח הסימולציה.
`victory_points` כולל קלפי נקודות מוסתרים; `public_victory_points` הוא הניקוד הגלוי.
משחק שנעצר במגבלה נשמר עם `winner: null` וסיבה `turn_limit` או `action_limit`.
זה אינו ניצחון או תיקו שנקבע לפי הניקוד. קוד יציאה 0 מציין שההרצה הושלמה,
גם אם חלק מהמשחקים נעצרו במגבלה.

לשחזור, משתמשים באותו seed, הגדרות וגרסאות נעולות. השחקן `SimulationRandomPlayer`
בוחר באופן אחיד מרשימת פעולות מסודרת, כדי לנטרל הבדלים בסדר קבוצות בין תהליכי Python.
מזהי המשחק, חותמות הזמן וזמני הריצה משתנים בין הרצות. תאימות זהה בין גרסאות Python
או בין גרסאות מנוע שונות אינה מובטחת.

## שימוש מתוך Python

```python
from catanzero.simulation import SimulationConfig, create_game, run_game

game, result = run_game(SimulationConfig(), seed=42)
print(result.winner, result.victory_points)

# הרצה צעד אחר צעד:
game = create_game(SimulationConfig(), seed=42)
record = game.play_tick()
print(record)
print(game.playable_actions)
```

טעינת מצב שמור והמשך סימולציה:

```python
import json
from pathlib import Path
from catanatron import Color
from catanatron.serialization import state_from_json
from catanzero.simulation import SimulationRandomPlayer

document = json.loads(Path("runs/my-run/game_0000.json").read_text(encoding="utf-8"))
players = [SimulationRandomPlayer(Color[color]) for color in document["colors"]]
game = state_from_json(document, players)
if game.winning_color() is None:
    record = game.play_tick()
```

## בדיקות ומבנה

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

הבדיקות מכסות ניצחון במשחק מלא, מגבלות עצירה, שחזור בין תהליכים בעלי hash seed שונה,
טעינה והמשך זהים מ־JSON, שמירת סדרת משחקים, מניעת דריסת תוצאות וקלט לא תקין.

```text
src/catanzero/simulation.py   API של יצירת משחק, הרצה וסיכום
src/catanzero/cli.py          פקודת הרצה ושמירת תוצאות
src/catanzero/viewer.py       שחזור המהלכים וייצוא תצוגת HTML
src/catanzero/replay.html     תצוגת הלוח ובקרי הניגון
tests/test_simulation.py     בדיקות אינטגרציה
pyproject.toml / uv.lock     חבילה ותלויות נעולות
run.cmd                     קיצור הרצה ל־Windows
runs/                       תוצאות מקומיות, לא לניהול גרסאות
```

## דאטה ממהלכי AB:2

המרת משחקי ההשוואה השמורים לדוגמאות אימון, עם שמונה דגימות מידע נסתר לכל החלטה:

```powershell
.\.venv\Scripts\python.exe -m catanzero.tournament_dataset runs/bot-comparison-12 --output runs/quality-AB2-12 --belief-samples 8 --workers 4
.\.venv\Scripts\python.exe tools/audit_quality_dataset.py runs/quality-AB2-12
```

בקובץ `candidate.moves.jsonl.gz` נשמרות החלטות AB:2 בלבד. יתר המהלכים, הלוחות,
והמשחקים המקוריים זמינים באותה תיקייה. הציון משקלל 50% היוריסטיקה ו־50% תוצאה
סופית, ואינו הסתברות מכוילת לנכונות המהלך. הוראות שימוש ומגבלות:
[דאטה ממהלכי AB:2](docs/quality-dataset.md).

## רגרסיה ליניארית עם קלט מנורמל

נוספה הכנה של 164 תכונות מספריות לפני המהלך ופלט משוקלל אחד, עם נרמול
לפי האימון בלבד והפרדה בין משחקי אימון, ולידציה ובדיקה. המודל הראשוני הוא
רגרסיה ליניארית Ridge. הוראות, בדיקות דליפת מידע ומגבלות:
[רגרסיה ליניארית](docs/linear-regression.md).

נבדקו גם עשר תצורות של שישה סוגי מודלים קלאסיים. הבחירה נעשתה לפי הוולידציה,
והתוצאות בבדיקה מעורבות: [השוואת המודלים](docs/model-comparison.md).

## ניסוי תיוגים ב־MLflow

נשמרו שני מודלי Gradient Boosting: תיוג 50% היוריסטיקה ו־50% תוצאה סופית,
מול תוצאה סופית בלבד. בוצעו משחקים מול F ומשחקים ישירים, ונשמר תיעוד מלא:
[ניסוי התיוגים מ־14 באוקטובר 2026](docs/label-target-experiment-2026-10-14.md).
המעקב המקומי זמין ב־[MLflow](http://127.0.0.1:5070/).

## השוואה ללא סימולציות קדימה

להשוואת מדיניות שבוחרת רק מהמצב הנוכחי, משתמשים בפרוטוקול החדש:

```powershell
.\.ml-venv\Scripts\python.exe -u tools/no_lookahead_experiment.py --output runs/no-lookahead-v2
```

כל המשתתפים מקבלים תצפית ורשימת פעולות בלבד. קריאות להעתקה, יצירה או התקדמות
של משחק נחסמות בזמן הבחירה. `F_static` הוא בוט יוריסטי חדש ללא חיפוש;
F ו־AB המקוריים אינם משתתפים בפרוטוקול זה. תוצאות הפיילוט ותיעוד:
[השוואה ללא חיפוש](docs/no-lookahead-experiment-2026-10-14.md).

## למידה מקוונת עם SARSA

נוסף מודל SGDRegressor ליניארי עם 165 פרמטרים שמתחילים מאפס. המודל עצמו משחק
מול שלושה F_static בשמונה סביבות מקבילות, ועדכון המשקלים מרוכז בתהליך אחד
אחרי כ־32 מעברים חדשים. בדיקות משחק נפרדות נעשות עם משקלים קפואים.
המודל והנרמול נשמרים ב־MLflow. תוצאות הפיילוט, אימות העדכונים והוראות שחזור:
[ניסוי Online SARSA](docs/online-sarsa-2026-10-14.md).

אפשר להמשיך ממשקלים ומצב אופטימייזר שמורים עם `--resume-from`, ולבחור זרעים
חדשים דרך `--training-base-seed` ו־`--evaluation-base-seed`. לוחות כל ניסויי המקור
מוחרגים. [המשך אימון: עוד 500 משחקים](docs/online-sarsa-continued-500-v2.md)
לא הראה שיפור: 21/200 לפני לעומת 19/200 אחרי על אותם לוחות חדשים. שתי הגרסאות נשמרו.

[המשך עם חקירה מופחתת מ־10% ל־5%](docs/online-sarsa-low-exploration-500-v3.md):
עוד 500 משחקי אימון. בבדיקה חדשה על אותם 200 לוחות לפני ואחרי, שיעור הניצחון
עלה מ־20/200 ל־30/200 (10%→15%). ההפרש לא מובהק במבחן המזווג (p≈0.132),
ולכן לפי כלל ההמשך שנקבע מראש לא נפתח סבב נוסף. גרסה 3 נשמרה ב־MLflow.

המנוע הוא פרויקט חיצוני ברישיון GPL-3.0-or-later. קוד המנוע מותקן כתלות,
ואינו מועתק לקוד המקור של CatanZero. תיעוד נוסף: [Catanatron](https://docs.catanatron.com/).
