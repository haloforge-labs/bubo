//go:build windows

// Test the actual serialized policy details at the production Go boundary.
package main

import (
	"bubo/gui/supervisor"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

func main() {
	baseline, err := os.ReadFile(os.Args[1])
	if err != nil {
		panic(err)
	}
	var event map[string]json.RawMessage
	if err = json.Unmarshal(baseline, &event); err != nil {
		panic(err)
	}
	event["type"] = json.RawMessage(`"error"`)
	for _, name := range os.Args[2:] {
		raw, err := os.ReadFile(name)
		if err != nil {
			panic(err)
		}
		var details map[string]json.RawMessage
		if err = json.Unmarshal(raw, &details); err != nil {
			panic(err)
		}
		payload := map[string]json.RawMessage{"code": json.RawMessage(`"memory_insufficient"`), "component": json.RawMessage(`"serve"`), "stage": json.RawMessage(`"startup"`), "retryable": json.RawMessage(`false`), "remedy": details["message"]}
		for _, key := range []string{"needed_bytes", "available_bytes", "sessions", "context"} {
			payload[key] = details[key]
		}
		body, _ := json.Marshal(payload)
		event["payload"] = body
		wire, _ := json.Marshal(event)
		os.WriteFile(filepath.Join(filepath.Dir(name), "wire-"+filepath.Base(name)), append(wire, '\n'), 0644)
		parsed, err := supervisor.ParseEvent(wire)
		if err != nil {
			panic(err)
		}
		got := parsed.Value.(*supervisor.CLIError)
		if got.NeededBytes == nil || got.AvailableBytes == nil || got.Sessions == nil || got.Context == nil || *got.NeededBytes <= *got.AvailableBytes || *got.Sessions != 1 || *got.Context != 200000 {
			panic("missing/incorrect shortfall")
		}
		fmt.Printf("PASS %s: needed=%d available=%d sessions=%d context=%d\n", name, *got.NeededBytes, *got.AvailableBytes, *got.Sessions, *got.Context)
	}
}
