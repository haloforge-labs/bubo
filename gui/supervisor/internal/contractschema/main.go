// Command contractschema refreshes the embedded canonical CLI schema.
package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
)

func main() {
	if err := copySchema(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func copySchema() error {
	source := filepath.Join("..", "..", "tests", "cli", "contract", "event.schema.json")
	data, err := os.ReadFile(source)
	if err != nil {
		return err
	}
	const target = "contract.schema.json"
	current, err := os.ReadFile(target)
	if err == nil && bytes.Equal(current, data) {
		return nil
	}
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	return os.WriteFile(target, data, 0644)
}
