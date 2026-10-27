package supervisor

import (
	"encoding/json"
	"reflect"
)

var rawMessageType = reflect.TypeOf(json.RawMessage(nil))

// Filled returns v with every nil slice and map replaced by an empty one. The
// Wails layer applies it to every bound-method result and event payload, so no
// value reaches the frontend with a null where its TypeScript type says list.
func Filled[T any](v T) T {
	fillNilLists(reflect.ValueOf(&v).Elem())
	return v
}

// fillNilLists replaces nil slices and maps in a View with empty ones. Go
// marshals a nil slice as JSON null, but the generated TypeScript types declare
// these fields as arrays and objects, so a first-start View (before the CLI has
// answered) crashed the frontend on view.switches.find (owner, 2026-10-15).
// Fixtures always filled every list, so no test rendered a null one. Byte
// slices and json.RawMessage keep their meaning: null is a valid raw value.
func fillNilLists(v reflect.Value) {
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		if !v.IsNil() {
			fillNilLists(v.Elem())
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			fillNilLists(v.Field(i))
		}
	case reflect.Slice:
		if v.Type() == rawMessageType || v.Type().Elem().Kind() == reflect.Uint8 {
			return
		}
		if v.IsNil() {
			if v.CanSet() {
				v.Set(reflect.MakeSlice(v.Type(), 0, 0))
			}
			return
		}
		for i := 0; i < v.Len(); i++ {
			fillNilLists(v.Index(i))
		}
	case reflect.Map:
		if v.IsNil() && v.CanSet() {
			v.Set(reflect.MakeMap(v.Type()))
		}
	}
}
