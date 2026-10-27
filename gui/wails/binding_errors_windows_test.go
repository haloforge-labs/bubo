//go:build windows

package main

import (
	"os"
	"reflect"
	"regexp"
	"testing"

	"bubo/gui/supervisor"
)

func bindingErrorInventory(t *testing.T) []string {
	t.Helper()
	data, err := os.ReadFile("frontend/wailsjs/go/main/App.d.ts")
	if err != nil {
		t.Fatal(err)
	}
	declarations := regexp.MustCompile(`export function (\w+)\(`).FindAllStringSubmatch(string(data), -1)
	generated := map[string]bool{}
	for _, match := range declarations {
		generated[match[1]] = true
	}
	app := reflect.TypeOf(&App{})
	if len(generated) != app.NumMethod() {
		t.Fatalf("generated %d methods; App exports %d", len(generated), app.NumMethod())
	}
	errorType := reflect.TypeOf((*error)(nil)).Elem()
	var untyped []string
	for i := 0; i < app.NumMethod(); i++ {
		m := app.Method(i)
		if !generated[m.Name] {
			t.Fatal("ungenerated binding", m.Name)
		}
		if m.Name == "GetView" {
			if m.Type.NumOut() != 1 || m.Type.Out(0) != reflect.TypeOf(supervisor.View{}) {
				t.Fatal("read binding must return the typed View")
			}
			continue
		}
		if m.Type.NumOut() != 1 || m.Type.Out(0).Implements(errorType) || m.Type.Out(0).Kind() != reflect.Struct {
			untyped = append(untyped, m.Name)
			continue
		}
		field, ok := m.Type.Out(0).FieldByName("Error")
		if !ok || field.Type != reflect.TypeOf((*supervisor.CLIError)(nil)) || field.Tag.Get("json") != "error" {
			untyped = append(untyped, m.Name)
		}
	}
	t.Logf("generated bindings enumerated: %d; untyped action/error paths: %d", len(generated), len(untyped))
	return untyped
}

func TestBindingsTypedErrors(t *testing.T) {
	if untyped := bindingErrorInventory(t); len(untyped) != 0 {
		t.Fatalf("every action binding must return one typed result with CLIError: %v", untyped)
	}
}
