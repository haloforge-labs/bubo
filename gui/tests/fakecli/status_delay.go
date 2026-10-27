//go:build windows

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Exclusive markers assign the next delay even when helpers overlap. Marking
// before sleeping makes the next process warm after a timeout kills this one.
func delayStatus() {
	values := os.Getenv("BUBO_FAKE_STATUS_DELAYS")
	if values == "" {
		return
	}
	for index, value := range strings.Split(values, ",") {
		marker := filepath.Join(dir, fmt.Sprintf("status-delay-%d", index+1))
		file, err := os.OpenFile(marker, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if os.IsExist(err) {
			continue
		}
		if err != nil {
			panic(err)
		}
		file.Close()
		delay, err := time.ParseDuration(strings.TrimSpace(value))
		if err != nil {
			panic(err)
		}
		time.Sleep(delay)
		return
	}
}
