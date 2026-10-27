package supervisor

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestAdditiveEventStillValidatesEnvelope(t *testing.T) {
	for _, name := range []string{"valid", "missing-seq", "wrong-payload", "wrong-product", "empty-type", "long-type", "numeric-type", "malformed-time"} {
		t.Run(name, func(t *testing.T) {
			var event map[string]any
			if err := json.Unmarshal(fixture(t, "progress"), &event); err != nil {
				t.Fatal(err)
			}
			event["type"] = "future-event"
			switch name {
			case "missing-seq":
				delete(event, "seq")
			case "wrong-payload":
				event["payload"] = []any{}
			case "wrong-product":
				event["product"].(map[string]any)["name"] = 42
			case "empty-type":
				event["type"] = ""
			case "long-type":
				event["type"] = strings.Repeat("x", 257)
			case "numeric-type":
				event["type"] = 42
			case "malformed-time":
				event["time"] = "2026-02-30T00:00:00Z"
			}
			line, err := json.Marshal(event)
			if err != nil {
				t.Fatal(err)
			}
			parsed, err := ParseEvent(line)
			if name == "valid" {
				if err != nil || parsed.Type != "future-event" || parsed.Value != nil {
					t.Fatalf("additive event: %+v, %v", parsed, err)
				}
			} else {
				var protocol *ProtocolError
				if !errors.As(err, &protocol) {
					t.Fatalf("malformed additive event: %v", err)
				}
			}
		})
	}
}

func TestKnownPayloadEnumIsNeverAdditive(t *testing.T) {
	line := bytes.Replace(fixture(t, "error-invalid_arguments"), []byte(`"invalid_arguments"`), []byte(`"future-error-code"`), 1)
	if _, err := ParseEvent(line); err == nil {
		t.Fatal("known error payload enum failure was ignored")
	}
}

func TestParserLimitsBeforeSchemaValidation(t *testing.T) {
	base := bytes.TrimSpace(fixture(t, "progress"))
	for name, extra := range map[string]string{
		"size":         `"future":"` + strings.Repeat("x", 1<<20) + `"`,
		"depth":        `"future":` + strings.Repeat("[", 129) + "null" + strings.Repeat("]", 129),
		"nested-keys":  `"future":{"key":1,"key":2}`,
		"escaped-keys": `"future":{"key":1,"\u006bey":2}`,
	} {
		t.Run(name, func(t *testing.T) {
			line := append(append([]byte{}, base[:len(base)-1]...), []byte(","+extra+"}")...)
			var protocol *ProtocolError
			if _, err := ParseEvent(line); !errors.As(err, &protocol) {
				t.Fatalf("parser limit lost: %v", err)
			}
		})
	}
}
