//go:build windows

package supervisor

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"bubo/gui/updater"
	"golang.org/x/sys/windows"
)

type UpdateHealth struct {
	InstallDir string
	Token      string
	Deadline   time.Time
}

func NewUpdateHealth(installDir, token, deadline string) (*UpdateHealth, error) {
	if token == "" && deadline == "" {
		return nil, nil
	}
	t, err := time.Parse(time.RFC3339Nano, deadline)
	if err != nil || !time.Now().Before(t) || token == "" || len(token) > 256 || strings.ContainsAny(token, " \t\r\n\x00") || !filepath.IsAbs(installDir) {
		return nil, fmt.Errorf("invalid updater health token, package path or deadline")
	}
	return &UpdateHealth{InstallDir: installDir, Token: token, Deadline: t}, nil
}

func (h *UpdateHealth) Acknowledge(s Snapshot) error {
	if !time.Now().Before(h.Deadline) || !s.Initialized {
		return fmt.Errorf("GUI health initialization failed or exceeded its deadline")
	}
	f, err := os.CreateTemp(h.InstallDir, ".bubo-health-*")
	if err != nil {
		return err
	}
	name := f.Name()
	defer os.Remove(name)
	if _, err = f.Write(updater.HealthContent(h.Token)); err != nil {
		f.Close()
		return err
	}
	if err = f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	if !time.Now().Before(h.Deadline) {
		return fmt.Errorf("GUI health deadline expired before acknowledgment")
	}
	from, err := windows.UTF16PtrFromString(name)
	if err != nil {
		return err
	}
	to, err := windows.UTF16PtrFromString(filepath.Join(h.InstallDir, updater.HealthFileName))
	if err != nil {
		return err
	}
	return windows.MoveFileEx(from, to, windows.MOVEFILE_REPLACE_EXISTING|windows.MOVEFILE_WRITE_THROUGH)
}
