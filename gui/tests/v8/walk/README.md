# A3 Windows walk

Set `BUBO_TEST_EVIDENCE_DIR` to an absolute scratch folder outside the checkout.
The V8 harness derives build copies, reports and screenshots from this folder.
Go and Wails are resolved from PATH and the active Go module cache.

The walk launches the tagged Bubo test executable through A0's hidden desktop
controller and Job Object. It uses CDP on the compiled A3 port. Never run the
executable directly. `build.mjs` also verifies the pinned Wails patch hashes.

Run from the repository root on Windows, after `hidden-gui-probe.mjs` has made
the A3 loopback certificate:

```powershell
node gui/tests/v8/build.mjs --job A3 --port 9334
node gui/tests/v8/hidden-gui-probe.mjs --job A3 --port 9334
node gui/tests/v8/walk/run.mjs --job A3 --port 9334
```

A failed check makes the walk exit nonzero; retain the summary for diagnosis.
The summary, captures, console output, rerun command, and input hashes are in
`%BUBO_TEST_EVIDENCE_DIR%\v8\A3\walk`; screenshots are in
`%BUBO_TEST_EVIDENCE_DIR%\shots\v8-A3`. Set `$env:WALK_STATES='S4'` for a focused state.
Set `$env:WALK_CONTROL_ROUTE='About'` and
`$env:WALK_CONTROL_TEXT='Copy version'`, then run `control-only.mjs` for a
focused control check. Set `$env:WALK_BREAK='W2'`, then run `breaks.mjs` to
apply, build, test, and restore a scratch fault; it retains
its JSON evidence under `walk\break-W2.json`.
Set `$env:WALK_CASES='controls'` for a separate W2/W3/W11 pass across the
selected states. It writes `summary-controls.json` and does not repeat the
other oracle cases.
Set `$env:WALK_CASES='links'` to check page links and sidebar navigation
after an earlier full pass tested the buttons. It writes `summary-links.json`.
`retry-failures.mjs` reads the complete `summary.json` and replays only its
failed W2 controls in fresh state instances. It writes `W2-retries.json` and
keeps the baseline summary intact.

S1-S16 and S18 are live real-executable states. S17 is explicitly omitted
because the screen is covered by A1's render
sweep and recovery by A4's F14 updater test. The shared S16 updater fixture
is `../updaterfixture` and uses production updater package and manifest APIs.
All external services in the walk bind loopback; the external fake server in
S14 has its own Job Object.

Each state inventories enabled main buttons and links, and clicks them along
with the sidebar navigation. If an earlier
click removes a control, that control is replayed against a fresh isolated
instance of the same initial state. W2 accepts a changed screen or View (or
native exit); the G1 external-link exception accepts a captured HTTPS URL.
S6 is identified by `models.download.state=running`; its outer server state
can still be `Stopped`.
The offscreen “Skip to content” link is activated with Enter on the hidden
WebView, as its keyboard-first placement intercepts pointer clicks.
W3 records returned backend errors. W11 fails a state with no
clicked controls. Other cases compare the literal v8 oracle, screen, tray
capture, log files, link capture, request/settings behavior, and native close.
