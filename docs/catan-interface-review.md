# Catan interface references — 2026-10-15

Reviewed official published screenshots, rather than playing the current live
products. The references depict older versions: Catan Universe's media assets
are under a 2020 path; the Colonist Discord article was published in 2023.
Player-to-player trading is outside this review.

Sources:
- [Catan Universe official media](https://catanuniverse.com/en/media/),
  [examined screenshot](https://catanuniverse.com/wp-content/uploads/2020/06/1.png).
- [Colonist official Discord guide](https://blog.colonist.io/play-colonist-on-discord/),
  [examined screenshot](https://blog.colonist.io/content/images/2023/12/Screenshot-2023-12-06-at-12.00.35-AM.png).
- [Colonist illustrated base-game rules](https://colonist.io/catan-rules).

Local copies of the two examined images are in `runs/ui-reference/`. They are
reference material only and are not served or used as game assets.

## Observations and implications

| Visual feature | Published reference | CatanZero implication |
|---|---|---|
| Board style | Colonist uses a flat, saturated, illustrated board; Universe uses a perspective wooden table and dimensional pieces. | Keep the flat, automatically fitted board for phone readability. Perspective is an aesthetic choice, not a prerequisite. |
| Terrain identification | Both combine terrain colors with pictures. Number tokens are distinct from the tile art. | Already present on tiles; reuse small resource symbols in the hand, prices and log, alongside English names. |
| Number readability | Large number tokens, probability dots, red 6 and 8. | Already present; retain contrast as pieces accumulate. |
| Piece ownership | Roads and buildings use the player's color. Cities and settlements have different silhouettes. | Already present; strong road outlines were added previously. |
| Player summaries | Separate colored player panels display scores and card counts; road/army statistics have symbols. | Scores and counts exist; clearer public road/knight progress and award icons are useful next additions. |
| Action context | Colonist shows the active player's current task and a timer, with build actions along the bottom; Universe groups action buttons and dice at the lower right. | Keep one consistent action area and a clear turn indicator. A timer is optional in this local bot game. |
| Public history | Colonist's log combines text, player colors and small action/resource pictures. | Text log exists; icons could make recent production/build events easier to scan. |
| Stock information | Colonist's screenshot shows bank resource counts and remaining build pieces. | Optional compact indicators could explain resource shortages and piece limits. |
| Dice | Universe's screenshot uses two high-contrast pip faces in a fixed location. | Implemented larger pip faces and a numeric sum in the persistent lower board bar. |

Recommended next work, based on these observations: consistent resource icons
first, then clear feedback on which unblocked tiles produce after a roll. Tile
production feedback is a proposed improvement; it was not verified as behavior
in these static screenshots. Avoid covering the board or shrinking it to add
decoration.

## Implemented and checked now

Dice faces are 50px desktop / 42px narrow screens, replacing the previous
25px / 19px numeric boxes. Each face uses standard pip positions, with an
explicit total, special 7 label and accessible announcement. All six faces,
7/non-7 transitions, empty/reset state and actual HTTP engine roll values passed
application-code integration checks. The engine and model were unchanged.
No rendered browser or physical-phone layout verification was available.
