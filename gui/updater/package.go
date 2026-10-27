package updater

import (
	"debug/pe"
	"fmt"
	"os"
	"path/filepath"
)

// PackageLayout is the updater's authoritative release executable layout.
type PackageLayout struct {
	CLI    string `json:"cli"`
	GUI    string `json:"gui"`
	Helper string `json:"helper"`
}

func ReleasePackageLayout() PackageLayout {
	return PackageLayout{CLI: cliRelative, GUI: guiRelative, Helper: helperRelative}
}

// PackageExecutables returns a fresh list so callers cannot mutate the contract.
func PackageExecutables() []string { return []string{cliRelative, guiRelative, helperRelative} }

// CheckPackageExecutables inspects images without starting any executable.
func CheckPackageExecutables(folder string) error {
	for _, rel := range PackageExecutables() {
		name := filepath.Join(folder, filepath.FromSlash(rel))
		info, err := os.Lstat(name)
		if err != nil || !info.Mode().IsRegular() {
			return fmt.Errorf("package is missing regular file %s", rel)
		}
		p, err := pe.Open(name)
		if err != nil {
			return fmt.Errorf("%s: %w", rel, err)
		}
		header, ok := p.OptionalHeader.(*pe.OptionalHeader64)
		p.Close()
		if !ok || p.Machine != pe.IMAGE_FILE_MACHINE_AMD64 {
			return fmt.Errorf("%s: expected Windows x64 PE", rel)
		}
		if rel != cliRelative && header.Subsystem != pe.IMAGE_SUBSYSTEM_WINDOWS_GUI {
			return fmt.Errorf("%s: expected Windows GUI subsystem", rel)
		}
	}
	return nil
}
