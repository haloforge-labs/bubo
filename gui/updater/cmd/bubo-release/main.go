// bubo-release is offline release tooling, never included in the package.
package main

import (
	"bubo/gui/updater"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"
)

func main() {
	info := flag.Bool("PackageInfo", false, "print canonical package executable names")
	folder := flag.String("CheckFolder", "", "inspect required package executables")
	archive := flag.String("Zip", "", "built package ZIP")
	output := flag.String("Out", "", "output manifest JSON")
	notes := flag.String("Notes", "", "UTF-8 user-facing summary file (required, nonempty)")
	runtimeChanged := flag.Bool("RuntimeChanged", false, "whether engine payloads changed (required)")
	runtimeBytes := flag.Int64("RuntimeDownloadBytes", 0, "engine download bytes (required)")
	modelsChanged := flag.Bool("ModelsChanged", false, "whether model payloads changed (required)")
	modelsPrepare := flag.String("ModelsNeedsPrepare", "", "comma-separated model IDs; explicit empty value for none (required)")
	var opts updater.ReleaseOptions
	flag.StringVar(&opts.Version, "Version", "", "default: packaged version")
	flag.StringVar(&opts.Tag, "Tag", "", "default: v + packaged version")
	flag.StringVar(&opts.Commit, "Commit", "", "default: packaged GUI commit")
	flag.StringVar(&opts.MinUpdatableFrom, "MinUpdatableFrom", "", "minimum supported installed version (required)")
	released := flag.String("Released", "", "UTC RFC3339 release time (required)")
	flag.IntVar(&opts.InstallPathLength, "InstallPathLength", 60, "maximum install folder length for path budget")
	flag.Parse()
	check := func(err error) {
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	if *info {
		check(json.NewEncoder(os.Stdout).Encode(updater.ReleasePackageLayout()))
		return
	}
	if *folder != "" {
		check(updater.CheckPackageExecutables(*folder))
		fmt.Println("PASS: required x64 executables and GUI subsystems (3)")
		return
	}
	if *archive == "" || *output == "" || *notes == "" {
		check(fmt.Errorf("Zip, Out and Notes are required"))
	}
	provided := map[string]bool{}
	flag.Visit(func(f *flag.Flag) { provided[f.Name] = true })
	for _, name := range []string{"RuntimeChanged", "RuntimeDownloadBytes", "ModelsChanged", "ModelsNeedsPrepare"} {
		if !provided[name] {
			check(fmt.Errorf("%s is required; no release decision is inferred", name))
		}
	}
	opts.RuntimeChanged, opts.RuntimeDownloadBytes, opts.ModelsChanged = runtimeChanged, runtimeBytes, modelsChanged
	opts.ModelsNeedsPrepare = []string{}
	if *modelsPrepare != "" {
		opts.ModelsNeedsPrepare = strings.Split(*modelsPrepare, ",")
	}
	text, err := os.ReadFile(*notes)
	check(err)
	opts.Notes = string(text)
	opts.Released, err = time.Parse(time.RFC3339, *released)
	check(err)
	m, longest, err := updater.GenerateReleaseManifest(*archive, opts)
	check(err)
	data, err := json.MarshalIndent(m, "", "  ")
	check(err)
	check(os.WriteFile(*output, append(data, '\n'), 0600))
	fmt.Printf("PASS: generated release manifest; zip_bytes=%d unpacked_bytes=%d longest_relative_path=%s install_path_budget=%d\n", m.Assets[0].Size, m.Assets[0].UnpackedBytes, longest, opts.InstallPathLength)
}
