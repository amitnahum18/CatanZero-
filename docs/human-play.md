# CatanZero browser game

The public [GitHub Pages game](https://amitnahum18.github.io/CatanZero-/) runs
the same rules and frozen Model 402 inside the browser. See
[deployment instructions](github-pages.md) for its device-local saves and build.
The instructions below describe the optional localhost server.

Run `play.cmd` from the project directory, then open http://127.0.0.1:8787/.
The server runs on this computer only. Leave its terminal open while playing;
Ctrl+C stops it. Use `play.cmd --port 8788` for a different port.

## Opponents

One human plays against 1–3 bots. Choose only the number of bots; every new
opponent uses frozen Model 402 (114 LightGBM ranking trees plus 64 ExtraTrees
trees, correction weight 0.2). There is no difficulty selector.
The model is not updated during games. Heuristic implementations remain in
the code for benchmarks, and legacy saved matches keep their original bots.

## Main screen

The interface is English. During play it shows the whole board and its ports,
player scores and card counts, your resources, all four build/buy options with
resource prices, bank trade controls, development cards and the latest action.
Unavailable purchases remain visible and disabled. Missing resources are marked
in the price. Resource gains and losses are shown beside your resource counts.

The last roll has two high-contrast pip dice (50px desktop, 42px narrow screens)
and a separate numeric total. A roll of 7 is marked `Robber roll`. It remains
labeled `Last roll` when the turn changes, and resets to `No roll yet` for a new
match. The display uses only actual server dice values. See
[interface reference review](catan-interface-review.md) for visual comparisons.

The SVG view fits all tiles, building intersections and compact port markers.
There are no zoom controls, panning gestures or mobile action drawers. On narrow
portrait screens the dashboard sits below the board; in landscape it sits beside
it. Older activity records are available in the activity log. Only rules and
confirmation of a new match use dialogs.

Select Road, Settlement or City, select a highlighted position, then confirm.
Selection alone does not change the game. Cancel discards the preview. Nearby
touches snap to the closest eligible intersection or road. Keyboard users can
focus marked locations and select them with Enter or Space.

Bank trade controls show the current 4:1, 3:1 or 2:1 ratio, based on the ports
owned by your settlements and cities. They submit only a currently legal trade.
Trading starts after rolling the dice. Before the roll, the button says
`Roll first`; the status also explains setup, discard, robber, insufficient-hand
and empty-bank restrictions. A bank trade requires the stated number of cards
of the same resource. The Give selector includes your current card counts.
Choosing a receive resource does not silently change the give resource.
Development cards requiring a resource choice show their choices directly in
the dashboard. Discard and robber-victim choices appear during the relevant
phase. Direct player-to-player trade is not included.

The engine enforces distance between settlements, connectivity, costs,
discarding, robber moves, development-card timing, Longest Road, Largest Army
and the 10-point victory condition. Two players use the simulator's variant on
the standard board. Responsive styling does not make this localhost server
available on a separate phone or publish it online.

## Saved games and private information

Each action saves the full server state to `runs/human-games/`. The browser
stores only an opaque match ID, so refreshing or restarting the server resumes
the match. Starting a new game keeps the previous match's file on disk.

The browser receives only a whitelisted player view: your private hand, opponents'
public scores/card counts and public board information. Hidden draws/thefts,
deck order, game seed and future random stream are not sent. Full save files are
not served over HTTP. Stale/duplicate actions are rejected using board revisions.
The server binds to localhost and rejects requests from other origins.

## Verification

- `tests/test_play_server.py`: twelve engine/HTTP checks, covering 2–4 players,
  placement distance, legal bot moves, original 402 predictions, hidden
  information, atomic-save retries, restoration, invalid/stale actions and a
  complete match to victory. Real HTTP bank/port tests verify 4:1, 3:1 and 2:1
  trades debit the correct cards, credit exactly one card, conserve bank stock,
  reject duplicate submissions and persist correctly. Controlled test positions
  also verify trading is unavailable before rolling.
- `node tests/check_play_frontend.cjs`: application-code integration using a test
  DOM and the actual HTTP server. Checks English copy, complete geometry fit,
  permanent purchase prices, preview/cancel/confirm, resource changes, 120 live
  decisions or a completed match, 2/4 players and saved-game restoration.
  Isolated view fixtures additionally check all six dice faces, the total,
  the 7 indicator, new-match reset, bank/port trade mapping, Monopoly,
  Year of Plenty and discards. Live rolls are checked against engine values.
  These fixtures do not submit artificial moves
  to the game engine.
- No browser is connected to the UI tool. The above checks are not visual
  validation of rendered layout, touch input on real hardware or phone screenshots.

The previous interface source is retained in `runs/play-web-table2-backup/`.

Roads have a thick dark outline and a wider player-color center. Legal road
placements use a yellow dashed center on a dark outline; selected placements
become a solid orange preview. Touch targets remain larger than the drawn pieces.
