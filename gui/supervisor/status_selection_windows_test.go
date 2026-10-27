//go:build windows

package supervisor

import (
	"reflect"
	"testing"
)

// Engine job E20, item 12: status reports readiness for the model the user
// selected. A manifest wins over a model file; neither means plain status.
func TestStatusSelectionArguments(t *testing.T) {
	s := &Supervisor{visibility: make(chan struct{}, 1)}
	if got := s.selectionArgs(); len(got) != 0 {
		t.Fatalf("nothing selected: %v", got)
	}
	s.SetModelSelection(`D:\a\bubo-package.json`, `D:\a\model.gguf`)
	if got := s.selectionArgs(); !reflect.DeepEqual(got, []string{"--manifest", `D:\a\bubo-package.json`}) {
		t.Fatalf("manifest: %v", got)
	}
	<-s.visibility // the change wakes the owner loop so the stream restarts
	s.SetModelSelection(`D:\a\bubo-package.json`, "")
	select {
	case <-s.visibility:
		t.Fatal("the same selection woke the loop")
	default:
	}
	s.SetModelSelection("", `D:\a\model.gguf`)
	if got := s.selectionArgs(); !reflect.DeepEqual(got, []string{"--model", `D:\a\model.gguf`}) {
		t.Fatalf("model: %v", got)
	}
	<-s.visibility
	s.SetModelSelection("", "")
	if got := s.selectionArgs(); len(got) != 0 {
		t.Fatalf("cleared: %v", got)
	}
}

func TestStatusTargetChangesWithTheSelection(t *testing.T) {
	a := statusTarget{enabled: true, selection: ""}
	b := statusTarget{enabled: true, selection: "--manifest\x00D:\a"}
	if a.matches(b) || !b.matches(b) {
		t.Fatal("a changed selection must restart the status stream")
	}
}
