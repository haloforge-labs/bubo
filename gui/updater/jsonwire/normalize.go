// Package jsonwire accepts the CLI's exact integral JSON numbers in the
// standard Go integer decoder, without rounding fractional values.
package jsonwire

import (
	"bytes"
	"encoding/json"
	"strconv"
	"strings"
)

// Unmarshal applies the shared CLI/package wire compatibility rule before the
// standard decoder. Its type, range and syntax checks remain in force.
func Unmarshal(data []byte, target any) error {
	return json.Unmarshal(Normalize(data), target)
}

// Normalize rewrites only integral number tokens with a decimal point or
// exponent and magnitude <= 2^53. All other bytes, including string contents,
// remain unchanged. Invalid JSON is returned untouched for the caller to reject.
// Decimal arithmetic avoids float64 rounding, underflow and overflow. The work
// and storage are bounded by the input size, even for enormous exponents.
func Normalize(data []byte) []byte {
	if !json.Valid(data) {
		return data
	}
	var out bytes.Buffer
	last := 0
	for i := 0; i < len(data); {
		switch c := data[i]; {
		case c == '"':
			i++
			for i < len(data) {
				if data[i] == '\\' {
					i += 2
				} else if data[i] == '"' {
					i++
					break
				} else {
					i++
				}
			}
		case c == '-' || c >= '0' && c <= '9':
			start := i
			for i < len(data) && (data[i] >= '0' && data[i] <= '9' || strings.ContainsRune("-+.eE", rune(data[i]))) {
				i++
			}
			token := string(data[start:i])
			if plain, ok := integer(token); ok {
				out.Write(data[last:start])
				out.WriteString(plain)
				last = i
			}
		default:
			i++
		}
	}
	if last == 0 {
		return data
	}
	out.Write(data[last:])
	return out.Bytes()
}

// integer receives a valid JSON number, so no syntax can be repaired here.
func integer(token string) (string, bool) {
	if !strings.ContainsAny(token, ".eE") {
		return "", false
	}
	negative := token[0] == '-'
	mantissa := strings.TrimPrefix(token, "-")
	exponent := "0"
	if i := strings.IndexAny(mantissa, "eE"); i >= 0 {
		exponent, mantissa = mantissa[i+1:], mantissa[:i]
	}
	fraction := 0
	if i := strings.IndexByte(mantissa, '.'); i >= 0 {
		fraction = len(mantissa) - i - 1
		mantissa = mantissa[:i] + mantissa[i+1:]
	}
	digits := strings.TrimLeft(mantissa, "0")
	if digits == "" {
		return "0", true
	}
	exp, err := strconv.ParseInt(exponent, 10, 64)
	// Beyond this bound a nonzero coefficient cannot be both integral and
	// within 2^53. Check before subtracting fraction or allocating padding.
	bound := int64(len(token)) + 16
	if err != nil || exp > bound || exp < -bound {
		return "", false
	}
	scale := exp - int64(fraction)
	if scale < 0 {
		if -scale >= int64(len(digits)) {
			return "", false
		}
		end := len(digits) + int(scale)
		if strings.Trim(digits[end:], "0") != "" {
			return "", false
		}
		digits = digits[:end]
	} else if scale > 0 {
		if int64(len(digits))+scale > 16 {
			return "", false
		}
		digits += strings.Repeat("0", int(scale))
	}
	const exactLimit = "9007199254740992"
	if len(digits) > len(exactLimit) || len(digits) == len(exactLimit) && digits > exactLimit {
		return "", false
	}
	if negative {
		digits = "-" + digits
	}
	return digits, true
}
