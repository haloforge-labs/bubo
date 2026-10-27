package jsonwire

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestNormalize(t *testing.T) {
	for _, tc := range []struct{ name, in, want string }{
		{"context", `2e+05`, `200000`},
		{"carve", `6.4e+10`, `64000000000`},
		{"weights", `8.779e+10`, `87790000000`},
		{"large", `1e+16`, `1e+16`},
		{"integral mantissa", `2.5e+05`, `250000`},
		{"fraction", `2.5e+00`, `2.5e+00`},
		{"negative", `-2e+05`, `-200000`},
		{"negative fraction", `-2.5`, `-2.5`},
		{"decimal integer", `200000.000`, `200000`},
		{"boundary", `9.007199254740992e+15`, `9007199254740992`},
		{"negative boundary", `-9007199254740992.0`, `-9007199254740992`},
		{"above boundary", `9007199254740993.0`, `9007199254740993.0`},
		{"below negative boundary", `-9007199254740993e0`, `-9007199254740993e0`},
		{"rounding fraction", `9007199254740991.5`, `9007199254740991.5`},
		{"tiny fraction", `1.00000000000000000001`, `1.00000000000000000001`},
		{"underflow", `1e-9999999999999999999999`, `1e-9999999999999999999999`},
		{"overflow", `1e9999999999999999999999`, `1e9999999999999999999999`},
		{"zero huge exponent", `-0e9999999999999999999999`, `0`},
		{"zero", `0.00e-99`, `0`},
		{"negative exponent integral", `1000e-3`, `1`},
		{"negative exponent fraction", `1001e-3`, `1001e-3`},
		{"strings", ` {"2e+05":"2e+05 \\\" 6.4e+10", "nested":[-2e+05,[8.779e+10,2.5]]} `, ` {"2e+05":"2e+05 \\\" 6.4e+10", "nested":[-200000,[87790000000,2.5]]} `},
		{"plain", ` {"n":200000,"s":"200000"} `, ` {"n":200000,"s":"200000"} `},
		{"invalid leading zero", `{"n":01e0}`, `{"n":01e0}`},
		{"invalid trailing dot", `1.`, `1.`},
		{"trailing value", `1e0 2e0`, `1e0 2e0`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input := []byte(tc.in)
			got := Normalize(input)
			if string(got) != tc.want {
				t.Fatalf("got %s, want %s", got, tc.want)
			}
			if string(input) != tc.in {
				t.Fatal("input mutated")
			}
			if !bytes.Equal(Normalize(got), got) {
				t.Fatal("not idempotent")
			}
		})
	}
}

func TestIntegerContract(t *testing.T) {
	for _, token := range []string{"2e+05", "6.4e+10", "8.779e+10", "2.5e+05"} {
		t.Run(token, func(t *testing.T) {
			var n uint64
			if err := Unmarshal([]byte(token), &n); err != nil {
				t.Fatal(err)
			}
		})
	}
	for _, token := range []string{"2.5e+00", "-2e+05", "1e+16", "1.00000000000000000001", "9007199254740991.5", "01e0", "1e0 2e0"} {
		t.Run("reject "+token, func(t *testing.T) {
			var n uint64
			if err := Unmarshal([]byte(token), &n); err == nil {
				t.Fatal("integer contract weakened")
			}
		})
	}
	var n int64
	if err := Unmarshal([]byte(`-2e+05`), &n); err != nil || n != -200000 {
		t.Fatal(n, err)
	}
	// Rewriting cannot bypass the original decoder's syntax validation.
	for _, raw := range []string{`{"x":01e0}`, `1e0 2e0`, `1.e0`} {
		var v any
		if json.Unmarshal([]byte(raw), &v) == nil {
			t.Fatal("invalid test case")
		}
		if Unmarshal([]byte(raw), &v) == nil {
			t.Fatal("invalid JSON accepted")
		}
	}
}
