// bubo-update must ship as bin/bubo-update.exe and be built with
// -ldflags=-H=windowsgui. Install copies it outside the install folder before use.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"bubo/gui/updater"
)

func main() {
	var o updater.HelperOptions
	flag.StringVar(&o.InstallDir, "install", "", "absolute install folder")
	flag.StringVar(&o.StagedDir, "staged", "", "absolute same-volume transaction folder")
	flag.StringVar(&o.WaitPIDs, "wait-pids", "", "exact PIDs, optionally PID:creation-filetime, comma separated")
	flag.DurationVar(&o.HealthTimeout, "health-timeout", updater.DefaultHealthTimeout, "new GUI health deadline")
	flag.DurationVar(&o.WaitTimeout, "wait-timeout", updater.DefaultWaitTimeout, "process exit deadline")
	flag.StringVar(&o.Version, "version", "", "staged product version")
	flag.StringVar(&o.Seal, "seal", "", "verified staged tree SHA-256")
	flag.BoolVar(&o.Recover, "recover", false, "resume interrupted rollback or committed cleanup")
	flag.Parse()
	if flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "unexpected positional arguments")
		os.Exit(2)
	}
	r, f := updater.RunHelper(context.Background(), o)
	if f != nil {
		fmt.Fprintln(os.Stderr, f)
		os.Exit(1)
	}
	if r.Status != "installed" && r.Status != "rolled_back" {
		os.Exit(1)
	}
}
