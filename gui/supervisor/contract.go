package supervisor

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"time"
	"unicode/utf8"

	"bubo/gui/updater/jsonwire"

	"github.com/santhosh-tekuri/jsonschema/v6"
	"github.com/santhosh-tekuri/jsonschema/v6/kind"
)

// The embedded file is copied byte for byte from the canonical CLI schema.
// Hand DTOs stay fixture-checked: the C2 generator spike could not preserve the
// frozen View without interface{} unions. See ARCHITECTURE.md section 3.
//
//go:generate go run ./internal/contractschema
//go:embed contract.schema.json
var contractSchema []byte
var eventSchema *jsonschema.Schema

func init() {
	var err error
	eventSchema, err = compileSchema(contractSchema)
	if err != nil {
		panic(err)
	}
}

// contractSchemaURL names the embedded schema. It must be absolute and must not
// use the file scheme. The compiler turns a relative name such as
// "contract.json" into a file URL built from the working directory, and it
// registers the resource under one spelling but looks it up under another when
// that directory contains a space, a non-ASCII letter or a percent sign. The
// lookup then missed, the compiler tried to open the file, and init() panicked
// before main, so the GUI exited with code 2 from such an install folder.
const contractSchemaURL = "https://schemas.bubo.invalid/gui/contract.schema.json"

func compileSchema(data []byte) (*jsonschema.Schema, error) {
	doc, err := jsonschema.UnmarshalJSON(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	c := jsonschema.NewCompiler()
	if err := c.AddResource(contractSchemaURL, doc); err != nil {
		return nil, err
	}
	return c.Compile(contractSchemaURL)
}

// The fixture inventory enumerates known events, while the reader's existing
// forward-compatibility rule accepts unknown names. Only the root type enum
// failure may be ignored; every envelope and payload constraint still runs in
// the library. This does not change or weaken the canonical fixture schema.
func additiveEventOnly(err *jsonschema.ValidationError) bool {
	if len(err.Causes) != 0 {
		for _, cause := range err.Causes {
			if !additiveEventOnly(cause) {
				return false
			}
		}
		return true
	}
	_, enum := err.ErrorKind.(*kind.Enum)
	return enum && len(err.InstanceLocation) == 1 && err.InstanceLocation[0] == "type"
}

type ProtocolError struct{ Detail string }

func (e *ProtocolError) Error() string { return "Protocol error: " + e.Detail }
func protocol(s string) error          { return &ProtocolError{s} }

func ParseEvent(line []byte) (Event, error) {
	var e Event
	// The schema validator consumes decoded JSON. Keep framing limits, UTF-8,
	// nesting and duplicate-key checks here: validation cannot recover those
	// properties after decoding (encoding/json otherwise keeps the last key).
	if len(line) > 1<<20 || !utf8.Valid(line) {
		return e, protocol("record size or UTF-8")
	}
	// Normalize the complete record once, before schema/envelope/payload
	// decoding. RawMessage fields and subsequent projections inherit it too.
	line = jsonwire.Normalize(line)
	dec := json.NewDecoder(bytes.NewReader(line))
	dec.UseNumber()
	value, err := readJSON(dec, 0)
	if err != nil {
		return e, protocol(err.Error())
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		return e, protocol("extra JSON value")
	}
	if o, ok := value.(map[string]any); ok {
		if v, present := o["schema_version"]; present {
			if n, isNum := v.(json.Number); isNum && n.String() != "1" {
				return e, protocol("unsupported schema_version " + n.String() + "; this window reads version 1")
			}
		}
	}
	if err := eventSchema.Validate(value); err != nil {
		if failure, ok := err.(*jsonschema.ValidationError); !ok || !additiveEventOnly(failure) {
			return e, protocol(err.Error())
		}
	}
	if kind, ok := value.(map[string]any)["type"].(string); !ok || kind == "" || len(kind) > 256 {
		return e, protocol("event type must be a nonempty string")
	}
	if err := json.Unmarshal(line, &e); err != nil {
		return e, protocol(err.Error())
	}
	if _, err := time.Parse("2006-01-02T15:04:05Z", e.Time); err != nil {
		return e, protocol("invalid UTC time")
	}
	var target any
	switch e.Type {
	case "capabilities":
		target = &Capabilities{}
	case "preflight":
		target = &Preflight{}
	case "plan":
		target = &Plan{}
	case "approval_required":
		target = &Approval{}
	case "progress":
		target = &Progress{}
	case "result":
		target = &Result{}
	case "error":
		target = &CLIError{}
	case "diagnose":
		target = &Diagnose{}
	case "diagnose_artifact":
		target = &Artifact{}
	case "status":
		target = &Status{}
	case "starting", "loading", "ready", "warning", "stopping", "stopped":
		target = &Lifecycle{}
	case "switches":
		target = &Switches{}
	case "request_completed":
		target = &RequestRow{}
	case "support_bundle":
		target = &SupportBundle{}
	case "models":
		target = &Models{}
	case "model_estimate":
		target = &ModelEstimate{}
	default:
		return e, nil // Additive events never drive state.
	}
	if err := json.Unmarshal(e.Payload, target); err != nil {
		return e, protocol(err.Error())
	}
	e.Value = target
	if m, ok := target.(*Models); ok {
		if err := m.check(); err != nil {
			return e, err
		}
	}
	if m, ok := target.(*ModelEstimate); ok {
		if err := m.check(); err != nil {
			return e, err
		}
	}
	if p, ok := target.(*Progress); ok && p.Total != nil && (p.Done > *p.Total || p.ResumeOffset > *p.Total) {
		return e, protocol("progress exceeds total")
	}
	if p, ok := target.(*Plan); ok {
		if p.Part >= p.Parts {
			return e, protocol("plan part is outside parts range")
		}
		for _, l := range p.Licenses {
			if l.TextOffset > l.TextBytes || uint64(len([]byte(l.Text))) > l.TextBytes-l.TextOffset {
				return e, protocol("license chunk exceeds text_bytes")
			}
		}
	}
	if l, ok := target.(*Lifecycle); ok && e.InstanceID != "" {
		if l.Identity != (Identity{e.InstanceID, e.PID, e.CreationTime, e.ExePath}) {
			return e, protocol("event identity and payload identity disagree")
		}
	}
	if st, ok := target.(*Status); ok {
		if _, err := time.Parse("2006-01-02T15:04:05Z", st.SampleTime); err != nil {
			return e, protocol("invalid status sample time")
		}
		for _, i := range st.Instances {
			if i.Identity == nil || *i.Identity != i.asIdentity() {
				return e, protocol("instance identity and flat identity disagree")
			}
		}
	}
	return e, nil
}

func readJSON(d *json.Decoder, depth int) (any, error) {
	if depth > 128 {
		return nil, fmt.Errorf("JSON nesting exceeds limit")
	}
	token, err := d.Token()
	if err != nil {
		return nil, err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return token, nil
	}
	switch delim {
	case '{':
		o := map[string]any{}
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return nil, err
			}
			k, ok := key.(string)
			if !ok {
				return nil, fmt.Errorf("invalid object key")
			}
			if _, exists := o[k]; exists {
				return nil, fmt.Errorf("duplicate key %s", k)
			}
			v, err := readJSON(d, depth+1)
			if err != nil {
				return nil, err
			}
			o[k] = v
		}
		_, err = d.Token()
		return o, err
	case '[':
		a := []any{}
		for d.More() {
			v, err := readJSON(d, depth+1)
			if err != nil {
				return nil, err
			}
			a = append(a, v)
		}
		_, err = d.Token()
		return a, err
	default:
		return nil, fmt.Errorf("unexpected delimiter")
	}
}

type CounterBaseline struct {
	Instance string
	Values   map[string]uint64
}

// Delta returns no deltas when the instance changes or any counter resets.
func (b *CounterBaseline) Delta(id string, values map[string]uint64) (map[string]uint64, bool) {
	reset := b.Instance != id || b.Values == nil
	for k, v := range values {
		old, ok := b.Values[k]
		if !ok || v < old {
			reset = true
		}
	}
	delta := map[string]uint64{}
	if !reset {
		for k, v := range values {
			delta[k] = v - b.Values[k]
		}
	}
	b.Instance = id
	b.Values = map[string]uint64{}
	for k, v := range values {
		b.Values[k] = v
	}
	return delta, reset
}
