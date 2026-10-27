# GUI telemetry

The GUI window uses Go and Wails with WebView2 and the site's CSS. The Go
supervisor and CLI client share the same contract. Wails handles tray actions
and its headless menu tests; the real tray check requires an interactive session.

The control panel reads the version 1 CLI status contract. Profile resolution,
memory planning and runtime counters stay in the CLI. A null value is
unavailable. `/health` alone does not prove token progress.

The server already exports Prometheus text at `/metrics` and session state at
`/slots`. To use Grafana, configure Prometheus to scrape the server's chosen
loopback address and port, with `/metrics` as the path. Then add that
Prometheus instance as a Grafana data source. This adds no Bubo endpoint.
Use the actual bound address and port from CLI status. For example, a server
at `http://127.0.0.1:8080/v1` has its metrics at
`http://127.0.0.1:8080/metrics`. These are examples, not a running service.

`RuntimeStatusMetrics` and `LlamaMetrics` read the same atomics in
`src/cli/serve/generation_metrics.hpp`. The contract has no second token
counter. See the audit in `docs/METRICS-CONTRACT.md` for names and gaps.
Live wall throughput uses monotonic sample time and token counter deltas.
Completed-request gauges use the request's recorded phase durations. Those
rates measure different intervals.

GUI controls come from registry rows with `gui_control: true`. `audience`
places them in Main, Advanced, or the collapsed Debug subsection of Advanced.
The subsection states that its switches are for troubleshooting. The four
existing budget/force-close diagnostics have `gui_control: false`; do not
turn them into GUI controls. The CLI diagnostic listing remains complete.

GPU watts must come from a named ADLX source and be labelled GPU power.
The status contract supplies no GPU watt reading or power write. The separate
supervisor follow-up owns the no-token-progress warning. It warns and offers
Restart; it must not kill a process on its own.
