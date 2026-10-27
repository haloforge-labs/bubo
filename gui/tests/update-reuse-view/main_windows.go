//go:build windows

// The real merged CLI's offline legacy-layout recordings, through the public
// supervisor ViewBuilder. Retain the View as the repeatable update-card input.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"bubo/gui/supervisor"
	"bubo/gui/updater"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	recordings := flag.String("recordings", "scenarios/recordings/release-update", "CLI capture directory")
	output := flag.String("output", "", "evidence View path")
	flag.Parse()
	if *output == "" {
		return fmt.Errorf("specify -output")
	}
	raw, err := os.ReadFile(filepath.Join(*recordings, "bubo-release.json"))
	if err != nil {
		return err
	}
	var release updater.Manifest
	if err := json.Unmarshal(raw, &release); err != nil {
		return err
	}
	in := supervisor.ViewInput{Source: "recorded", Snapshot: supervisor.Snapshot{Initialized: true, State: supervisor.Stopped, Prepared: true}, Update: supervisor.Update{State: supervisor.UpdateAvailable, Ready: true, CurrentVersion: "0.1.0", AvailableVersion: release.Version}}
	size := uint64(release.Assets[0].Size)
	in.Update.SizeBytes = &size
	for _, name := range []string{"legacy-runtime-preview.jsonl", "legacy-model-plan.jsonl", "legacy-status.json"} {
		file, err := os.Open(filepath.Join(*recordings, name))
		if err != nil {
			return err
		}
		scanner := bufio.NewScanner(file)
		scanner.Buffer(make([]byte, 65536), 4<<20)
		for scanner.Scan() {
			event, err := supervisor.ParseEvent(scanner.Bytes())
			if err != nil {
				file.Close()
				return fmt.Errorf("%s: %w", name, err)
			}
			switch value := event.Value.(type) {
			case *supervisor.Plan:
				in.Snapshot.PlanParts = append(in.Snapshot.PlanParts, *value)
			case *supervisor.Status:
				in.Snapshot.Status = value
			}
		}
		err = scanner.Err()
		file.Close()
		if err != nil {
			return err
		}
	}
	view := new(supervisor.ViewBuilder).Build(in)
	if view.PlanError != nil || len(view.Preparation) != 2 || view.Update.RuntimeDownloadBytes == nil || *view.Update.RuntimeDownloadBytes != 0 || view.Update.ModelsDownloadBytes == nil || *view.Update.ModelsDownloadBytes != 0 {
		return fmt.Errorf("legacy CLI reuse did not reach View: plans=%d error=%+v update=%+v", len(view.Preparation), view.PlanError, view.Update)
	}
	for _, plan := range view.Preparation {
		for _, file := range plan.Files {
			if file.State != "reuse" {
				return fmt.Errorf("%s/%s is %s", plan.Kind, file.FileID, file.State)
			}
		}
		fmt.Printf("%s: %d real payload sources, every state reuse\n", plan.Kind, len(plan.Files))
	}
	raw, err = json.MarshalIndent(view, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(*output), 0700); err != nil {
		return err
	}
	if err := os.WriteFile(*output, append(raw, '\n'), 0600); err != nil {
		return err
	}
	fmt.Printf("PASS: 0.1.0 -> %s; package %d bytes; engine 0 bytes; models 0 bytes; %s\n", release.Version, size, *output)
	return nil
}
