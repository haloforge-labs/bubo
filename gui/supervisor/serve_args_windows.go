//go:build windows

package supervisor

import "strconv"

type serveArgument struct {
	name       string
	value      string
	base       bool
	bare       bool
	reserved   bool
	preference string
}

// This is the sole ownership table for serve arguments and environment inputs.
// Override eligibility derives from it, including credential/runtime handoff.
func serveArguments(s Settings, host string, port int) []serveArgument {
	context, sessions, portValue := "", "", ""
	if s.Context != 0 {
		context = strconv.FormatUint(s.Context, 10)
	}
	if s.Sessions != 0 {
		sessions = strconv.FormatUint(uint64(s.Sessions), 10)
	}
	if port != 0 {
		portValue = strconv.Itoa(port)
	}
	return []serveArgument{
		{name: "--jsonl", base: true, bare: true},
		{name: "--host", value: host, base: true, preference: "lan_access"},
		{name: "--port", value: portValue, base: true, preference: "port"},
		{name: "--api-key", reserved: true}, // credentials use ServeKeyEnv
		{name: ServeKeyEnv, reserved: true},
		{name: RuntimeDirEnv, value: s.RuntimePath, reserved: true, preference: "runtime_path"},
		{name: "--profile", value: s.Profile, preference: "profile"},
		{name: "--context", value: context, preference: "context"},
		{name: "--sessions", value: sessions, preference: "sessions"},
		{name: "--model", value: s.ModelPath, preference: "model_path"},
		{name: "--manifest", value: s.ManifestPath, preference: "manifest_path"},
		{name: "--npu-prefill", value: s.NPUPrefill, preference: "npu_prefill"},
		{name: "--reasoning-effort", value: s.Effort, preference: "effort"},
	}
}

func buildServeArguments(s Settings, host string, port int, base bool) []string {
	var args []string
	for _, row := range serveArguments(s, host, port) {
		if row.base != base || row.reserved {
			continue
		}
		if row.bare {
			args = append(args, row.name)
		} else if row.value != "" {
			args = append(args, row.name, row.value)
		}
	}
	return args
}

func startArgs(s Settings) []string { return buildServeArguments(s, "", 0, false) }
