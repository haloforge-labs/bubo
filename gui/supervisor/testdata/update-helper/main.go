// A test seam for the shipped helper image: record the real updater's hidden
// process handoff without swapping a package or exiting the Go test process.
package main

import (
	"encoding/json"
	"os"
)

func main() {
	cwd, err := os.Getwd()
	if err != nil {
		panic(err)
	}
	body, err := json.Marshal(struct {
		Args []string
		CWD  string
		PID  int
	}{os.Args, cwd, os.Getpid()})
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(os.Getenv("BUBO_TEST_UPDATE_HANDOFF"), body, 0600); err != nil {
		panic(err)
	}
}
