// Offline adapter for go-licenses. Its report/save commands discover source
// URLs over non-registry HTTP, forbidden in the packaging lane. Libraries and
// its established classifier work locally; go list supplies exact build deps.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/google/go-licenses/licenses"
)

func main() {
	dir := flag.String("dir", "", "module to inspect")
	pkg := flag.String("package", ".", "binary package")
	licenseFile := flag.String("license", "", "classify a local vendored licence text")
	flag.Parse()
	check := func(err error) {
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	classifier, err := licenses.NewClassifier(0.9)
	check(err)
	if *licenseFile != "" {
		name, _, err := classifier.Identify(*licenseFile)
		check(err)
		fmt.Println(name)
		return
	}
	check(os.Chdir(*dir))
	libs, err := licenses.Libraries(context.Background(), classifier, false, []string{"bubo/gui"}, *pkg)
	check(err)
	type record struct{ Name, Version, License, LicensePath string }
	rows := []record{}
	for _, lib := range libs {
		license := "UNKNOWN"
		if lib.LicensePath != "" {
			name, _, err := classifier.Identify(lib.LicensePath)
			check(err)
			license = name
		}
		rows = append(rows, record{lib.Name(), lib.Version(), license, lib.LicensePath})
	}
	check(json.NewEncoder(os.Stdout).Encode(rows))
}
