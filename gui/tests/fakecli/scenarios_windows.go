//go:build windows

package main

import (
	"encoding/json"
	"fmt"
	"os"

	recordings "bubo/gui/tests/scenarios/replay"
)

// BUBO_FAKE_SCENARIO selects the shared recordings for one-shot reads. Dynamic
// serve/stop tests retain the fake's identity-checked lifecycle implementation.
func recordedScenario() bool {
	folder := os.Getenv("BUBO_FAKE_SCENARIO")
	if folder == "" {
		return false
	}
	kind := ""
	switch os.Args[1] {
	case "status":
		kind = "status"
	case "models":
		if len(os.Args) > 2 && os.Args[2] == "list" {
			kind = "models"
		}

	case "serve":
		if hasFlag("--list-switches") {
			kind = "switches"
		}
	case "prepare":
		// The recording is the engine plan; a catalogue model download keeps
		// the fake's own flow.
		if hasFlag("--plan") && flag("--catalogue-id") == "" {
			kind = "prepare"
		}
	}
	if kind == "" {
		return false
	}
	// A scenario answers only the commands it recorded.
	manifest, _, err := recordings.Read(folder)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if manifest.Records[kind] == "" {
		return false
	}
	if kind == "models" {
		_, status := recordedPayload(folder, "status")
		if readiness, ok := status["readiness"].(map[string]any); ok && readiness["runtime"] == false {
			runtimeMissingForModels()
		}
	}
	events, err := recordings.Command(folder, kind)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	for _, e := range events {
		b, err := json.Marshal(e)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		fmt.Println(string(b))
	}
	return true
}

// recordedPayload returns the first raw record of one kind in a scenario.
func recordedPayload(folder, kind string) (string, map[string]any) {
	_, records, err := recordings.Read(folder)
	if err == nil && len(records[kind]) == 0 {
		err = fmt.Errorf("scenario %s has no %s recording", folder, kind)
	}
	var payload map[string]any
	if err == nil {
		err = json.Unmarshal(records[kind][0].Payload, &payload)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	return records[kind][0].Type, payload
}
