//go:build windows

package supervisor

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/registry"
)

const loginRunKey = `Software\Microsoft\Windows\CurrentVersion\Run`
const loginRunValue = "Bubo.Control"

var loginKeyPath = loginRunKey
var loginStartupPath string

// SetLoginStart owns only Bubo's per-user Run value and the legacy shortcut.
// The command is the quoted GUI executable with no arguments.
func SetLoginStart(enabled bool, exe string) error {
	startup := loginStartupPath
	if startup == "" {
		var err error
		startup, err = windows.KnownFolderPath(windows.FOLDERID_Startup, 0)
		if err != nil {
			return err
		}
	}
	return setLoginStartAt(enabled, exe, registry.CURRENT_USER, loginKeyPath, startup)
}

// RestoreLoginStart reconciles the saved choice at startup, migrating an old
// shortcut even when the preference has not changed since upgrading.
func (b *NativeBackend) RestoreLoginStart() error {
	b.mu.RLock()
	enabled, exe := b.settings.StartGUIAtLogin, b.guiExe
	b.mu.RUnlock()
	return SetLoginStart(enabled, exe)
}

func setLoginStartAt(enabled bool, exe string, root registry.Key, keyPath, startup string) error {
	if enabled {
		if !filepath.IsAbs(exe) {
			return fmt.Errorf("GUI path must be absolute")
		}
		st, err := os.Stat(exe)
		if err != nil {
			return err
		}
		if st.IsDir() {
			return fmt.Errorf("GUI path must be a file")
		}
	}
	key, err := registry.OpenKey(root, keyPath, registry.QUERY_VALUE|registry.SET_VALUE)
	if errors.Is(err, registry.ErrNotExist) {
		if !enabled {
			return removeLegacyLoginShortcut(startup)
		}
		key, _, err = registry.CreateKey(root, keyPath, registry.QUERY_VALUE|registry.SET_VALUE)
	}
	if err != nil {
		return err
	}
	defer key.Close()
	old, _, readErr := key.GetStringValue(loginRunValue)
	if readErr != nil && !errors.Is(readErr, registry.ErrNotExist) {
		return readErr
	}
	if enabled {
		err = key.SetStringValue(loginRunValue, `"`+exe+`"`)
	} else {
		err = key.DeleteValue(loginRunValue)
		if errors.Is(err, registry.ErrNotExist) {
			err = nil
		}
	}
	if err != nil {
		return err
	}
	if err := removeLegacyLoginShortcut(startup); err != nil {
		// Restore only our own value if migration cannot remove the old launcher.
		var rollback error
		if readErr == nil {
			rollback = key.SetStringValue(loginRunValue, old)
		} else {
			rollback = key.DeleteValue(loginRunValue)
			if errors.Is(rollback, registry.ErrNotExist) {
				rollback = nil
			}
		}
		return errors.Join(err, rollback)
	}
	return nil
}

func removeLegacyLoginShortcut(startup string) error {
	err := os.Remove(filepath.Join(startup, "Bubo.Control.lnk"))
	if os.IsNotExist(err) {
		return nil
	}
	return err
}
