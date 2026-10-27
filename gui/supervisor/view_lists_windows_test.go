package supervisor

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// The owner's first launch (2026-10-15) crashed with "Cannot read properties of
// null (reading 'find')": a View built before the CLI answered carried null
// lists (switches, prepared_models, ...). Every list and map in the published
// View must be present, at any depth.
func TestViewNeverPublishesNullLists(t *testing.T) {
	for name, in := range map[string]ViewInput{
		"zero input":          {},
		"initialised no data": {Snapshot: Snapshot{Initialized: true, State: Stopped}},
		"failed first status": {Snapshot: Snapshot{Initialized: true, State: Failed, Label: "context deadline exceeded"}},
		"ready with status":   actionTestInput(Ready),
	} {
		v := new(ViewBuilder).Build(in)
		if nils := nilLists(reflect.ValueOf(v), "view"); len(nils) > 0 {
			t.Fatalf("%s: nil lists in the View: %v", name, nils)
		}
		data, err := json.Marshal(v)
		if err != nil {
			t.Fatal(name, err)
		}
		for _, field := range []string{"switches", "settings", "switch_overrides", "effective_switches", "prepared_models", "requests", "slots", "profiles"} {
			if strings.Contains(string(data), `"`+field+`":null`) {
				t.Fatalf("%s: %s is null in the published JSON", name, field)
			}
		}
	}
}

func nilLists(v reflect.Value, path string) []string {
	var out []string
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		if !v.IsNil() {
			out = append(out, nilLists(v.Elem(), path)...)
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if v.Type().Field(i).IsExported() {
				out = append(out, nilLists(v.Field(i), path+"."+v.Type().Field(i).Name)...)
			}
		}
	case reflect.Slice:
		if v.Type() == rawMessageType || v.Type().Elem().Kind() == reflect.Uint8 {
			return nil
		}
		if v.IsNil() {
			return []string{path}
		}
		for i := 0; i < v.Len(); i++ {
			out = append(out, nilLists(v.Index(i), path+"[]")...)
		}
	case reflect.Map:
		if v.IsNil() {
			return []string{path}
		}
	}
	return out
}
