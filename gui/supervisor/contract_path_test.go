package supervisor

import (
	"os"
	"path/filepath"
	"testing"
)

// The contract schema is compiled by init() before main. The compiler once
// derived the schema's address from the working directory, so a folder with a
// space or a non-ASCII letter made init() panic and the GUI exited with code 2
// before it opened. The schema must compile from any working directory.
func TestCompileSchemaFromAnyWorkingDirectory(t *testing.T) {
	for _, name := range []string{"Bubo Walk", "Bubo Wälk", "Bubo Wälk Test", "Jörg", "100%"} {
		t.Run(name, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), name)
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			t.Chdir(dir)
			if _, err := compileSchema(contractSchema); err != nil {
				t.Fatalf("compileSchema from %q: %v", dir, err)
			}
		})
	}
}
