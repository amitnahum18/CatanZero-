# ציון משוקלל למהלך

הציון משלב הערכה יוריסטית של המהלך ותוצאה סופית של השחקן שביצע אותו:

```text
weighted_score = h_weight * heuristic_score + outcome_weight * terminal_outcome
```

ברירת המחדל היא 0.5 לכל רכיב. המשקלים מנורמלים כך שסכומם 1; אפשר למשל
להזין 3 ו־7 כדי לקבל 30% יוריסטיקה ו־70% תוצאה. המשקלים קבועים לאורך המשחק.
אין שימוש במספר התור, באורך הסופי של המשחק או ביחס ביניהם לחישוב המשקל.

`heuristic_score` חייב להיות בטווח 0–1, בסולם קבוע ומתועד שבו ערך גבוה יותר
מציין מהלך טוב יותר. `terminal_outcome` הוא 1 עבור המנצח ו־0 עבור שאר השחקנים.
הסולם של היוריסטיקה משפיע על משמעות המשקלים גם כאשר כל הערכים בטווח 0–1.
הציון המשוקלל הוא יעד אימון לפי ההגדרה הזאת, ולא הסתברות ניצחון או מדד ביטחון.
גם מהלך חלש של מנצח מקבל את רכיב הניצחון; לכן שומרים את שני הרכיבים בנפרד.

משחק ארוך שהסתיים מקבל תיוג רגיל. במשחק שנעצר במגבלה, התוצאה הסופית והציון
המשוקלל הם `null`, והיוריסטיקה נשמרת. אם משקל התוצאה הוא 0 במפורש,
אפשר לקבל ציון יוריסטי בלבד גם במשחק שנעצר.

## שימוש

```python
from catanzero.scoring import ScoreConfig, score_move

targets = score_move(
    heuristic_score=0.8,
    actor="BLUE",
    winner="BLUE",
    termination="victory",
    config=ScoreConfig(heuristic_weight=0.5, outcome_weight=0.5),
)
assert targets["weighted_score"] == 0.9
```

אפשר לתייג קובץ JSONL עם שורה לכל מהלך. כל שורה דורשת את השדות
`actor`, `winner`, `termination`, `heuristic_score`. שדות נוספים נשמרים,
והתוצאות נכתבות תחת `targets`, כולל המשקלים וגרסת שיטת החישוב.
`winner`, `termination`, `heuristic_score` ו־`targets` הם מידע לתיוג בלבד;
בעת אימון משתמשים בקלט המודל מתוך `inputs`, בלי תוצאת המשחק או מידע עתידי.

```powershell
.\.venv\Scripts\python.exe -m catanzero.scoring examples\weighted_moves.jsonl --output runs\weighted-example.jsonl

# שינוי היחס ל־30% יוריסטיקה ו־70% תוצאה, ללא דריסת הקובץ הקודם:
.\.venv\Scripts\python.exe -m catanzero.scoring examples\weighted_moves.jsonl --output runs\weighted-example-30-70.jsonl --heuristic-weight 0.3 --outcome-weight 0.7
```

הדוגמאות בקובץ `examples/weighted_moves.jsonl` מומצאות ומסומנות `example_only`.
אין להשתמש בהן כדאטה אמיתי. המודול מקבל ציון יוריסטי שחושב מראש.
איסוף מצבים וחישוב יוריסטיקה אוטומטית זמינים כעת במודול `catanzero.dataset`;
ראו [תיעוד הפיילוט](dataset-pilot.md). לפני הרחבה יש לבדוק את היוריסטיקה.
