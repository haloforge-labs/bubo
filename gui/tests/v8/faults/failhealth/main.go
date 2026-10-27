//go:build windows

// failhealth stands in for a candidate GUI that starts and exits before it
// acknowledges the updater's health token. The helper must then roll back and
// restart the old GUI from the install folder.
package main

import "os"

func main() { os.Exit(3) }
