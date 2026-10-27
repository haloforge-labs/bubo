//go:build windows && bubo_test

package main

import (
	"bubo/gui/supervisor"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"time"

	"golang.org/x/sys/windows/registry"
)

var user32 = syscall.NewLazyDLL("user32.dll")

func emit(value string) {
	if path := os.Getenv("BUBO_TEST_HELPER_OUTPUT"); path != "" {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
		if err != nil {
			panic(err)
		}
		defer f.Close()
		if _, err := f.WriteString(value + "\n"); err != nil {
			panic(err)
		}
		return
	}
	fmt.Println(value)
}

func runPresent() bool {
	k, err := registry.OpenKey(registry.CURRENT_USER, `Software\Microsoft\Windows\CurrentVersion\Run`, registry.QUERY_VALUE)
	if err != nil {
		return false
	}
	defer k.Close()
	names, err := k.ReadValueNames(0)
	if err != nil {
		return false
	}
	for _, name := range names {
		if name == "Bubo.Control" {
			return true
		}
	}
	return false
}

func credentialPresent() bool {
	// cmdkey lists target names and metadata; it never prints credential blobs.
	output, err := exec.Command("cmdkey.exe", "/list").Output()
	return err == nil && strings.Contains(strings.ToLower(string(output)), "bubo/serve-api-key")
}

func observation() {
	seq, _, _ := user32.NewProc("GetClipboardSequenceNumber").Call()
	result := struct {
		ClipboardSequence uint32 `json:"clipboard_sequence"`
		OwnerRunPresent   bool   `json:"owner_run_present"`
		OwnerKeyPresent   bool   `json:"owner_key_present"`
	}{uint32(seq), runPresent(), credentialPresent()}
	data, err := json.Marshal(result)
	if err != nil {
		panic(err)
	}
	emit(string(data))
}

func decoyRegistry(op string) {
	c := supervisor.TestConfiguration()
	if op == "seed" {
		k, _, err := registry.CreateKey(registry.CURRENT_USER, c.LoginKeyPath, registry.ALL_ACCESS)
		if err != nil {
			panic(err)
		}
		defer k.Close()
		if err := k.SetStringValue("Bubo.Control", "A0 decoy sentinel"); err != nil {
			panic(err)
		}
		return
	}
	k, err := registry.OpenKey(registry.CURRENT_USER, c.LoginKeyPath, registry.ALL_ACCESS)
	if err != nil {
		panic(err)
	}
	defer k.Close()
	if op == "check" {
		v, _, err := k.GetStringValue("Bubo.Control")
		emit(fmt.Sprint(err == nil && v == "A0 decoy sentinel"))
		return
	}
	if op == "present" {
		_, _, err := k.GetStringValue("Bubo.Control")
		emit(fmt.Sprint(err == nil))
		return
	}
	if op == "value" {
		v, _, err := k.GetStringValue("Bubo.Control")
		if err != nil {
			panic(err)
		}
		emit(v)
		return
	}
	if op == "clean" {
		_ = k.DeleteValue("Bubo.Control")
		k.Close()
		_ = registry.DeleteKey(registry.CURRENT_USER, c.LoginKeyPath)
		return
	}
	panic("unknown registry command")
}

func decoyCredential(op string) {
	c := supervisor.TestConfiguration()
	store := supervisor.WindowsCredentialStore{}
	if op == "seed" {
		if err := store.Write(c.KeyTarget, "A0 decoy sentinel"); err != nil {
			panic(err)
		}
		return
	}
	if op == "check" {
		v, err := store.Read(c.KeyTarget)
		emit(fmt.Sprint(err == nil && v == "A0 decoy sentinel"))
		return
	}
	if op == "clean" {
		_ = store.Delete(c.KeyTarget)
		return
	}
	panic("unknown credential command")
}

func holdInstance() {
	lease, primary, err := supervisor.AcquireSingleInstance()
	if err != nil || !primary {
		panic(fmt.Sprintf("decoy instance lease: primary=%t err=%v", primary, err))
	}
	defer lease.Close()
	emit("READY")
	select {
	case <-lease.ShowWindow:
		emit("ACTIVATED")
	case <-time.After(10 * time.Minute):
	}
}

func main() {
	if len(os.Args) < 2 {
		panic("helper command required")
	}
	switch os.Args[1] {
	case "observe":
		observation()
	case "registry":
		decoyRegistry(os.Args[2])
	case "credential":
		decoyCredential(os.Args[2])
	case "holder":
		holdInstance()
	default:
		panic("unknown helper command")
	}
}
