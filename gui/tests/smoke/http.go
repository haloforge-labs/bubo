//go:build windows

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"bubo/gui/supervisor"
)

// The fixed test prompt. The first user turn is a fixed filler sentence repeated
// 40 times, then one short instruction. The filler makes the shared prefix long
// enough for a cache to reuse. The text is never printed: the transcript shows
// only its length and SHA-256.
const (
	promptFiller = "The quick brown fox jumps over the lazy dog. "
	promptAsk1   = "Reply with the single word: ready."
	promptAsk2   = "Now reply with the single word: done."
)

func promptFirstTurn() string { return strings.Repeat(promptFiller, 40) + promptAsk1 }

func promptLabel() string {
	p := promptFirstTurn()
	sum := sha256.Sum256([]byte(p))
	return fmt.Sprintf("fixed prompt, %d bytes, sha256 %s", len(p), hex.EncodeToString(sum[:])[:16])
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResult struct {
	HTTPStatus       int
	PromptTokens     int64
	CompletionTokens int64
	CachedTokens     int64
	TTFTms           float64 // Server-Timing ttft, -1 when absent
	DecodeTPS        float64 // timings.predicted_per_second, -1 when absent
	PrefillTPS       float64
	FinishReason     string
	ContentBytes     int
	Content          string // kept in memory for the next turn; never logged
	WallMS           float64
	ErrorCode        string // error.code or error.type only, never a message
}

var ttftPattern = regexp.MustCompile(`ttft;\s*dur=([0-9.]+)`)

// chat sends one non-streaming request: temperature 0, thinking off.
func chat(ctx context.Context, client *http.Client, endpoint, model string, messages []chatMessage, maxTokens int) (chatResult, error) {
	res := chatResult{TTFTms: -1, DecodeTPS: -1, PrefillTPS: -1}
	body, _ := json.Marshal(map[string]any{
		"model": model, "messages": messages, "max_tokens": maxTokens, "temperature": 0, "stream": false,
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(endpoint, "/")+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return res, err
	}
	req.Header.Set("Content-Type", "application/json")
	// Loopback serving needs no key. This is the public dummy key from the Pi snippet.
	req.Header.Set("Authorization", "Bearer "+supervisor.LoopbackDummyKey)
	t0 := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return res, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	res.WallMS = float64(time.Since(t0).Microseconds()) / 1000
	res.HTTPStatus = resp.StatusCode
	if err != nil {
		return res, err
	}
	var doc map[string]any
	if json.Unmarshal(raw, &doc) != nil {
		if resp.StatusCode == 200 {
			return res, fmt.Errorf("HTTP 200 body is not JSON (%d bytes)", len(raw))
		}
		return res, nil
	}
	if resp.StatusCode != 200 {
		if e, ok := doc["error"].(map[string]any); ok {
			for _, k := range []string{"code", "type"} {
				if s, ok := e[k].(string); ok && len(s) <= 64 && !strings.ContainsAny(s, " \r\n") {
					res.ErrorCode = s
					break
				}
			}
		}
		return res, nil
	}
	if m := ttftPattern.FindStringSubmatch(resp.Header.Get("Server-Timing")); m != nil {
		res.TTFTms, _ = strconv.ParseFloat(m[1], 64)
	}
	num := func(m map[string]any, key string) (float64, bool) {
		if m == nil {
			return 0, false
		}
		f, ok := m[key].(float64)
		return f, ok
	}
	sub := func(key string) map[string]any { m, _ := doc[key].(map[string]any); return m }
	usage := sub("usage")
	if f, ok := num(usage, "prompt_tokens"); ok {
		res.PromptTokens = int64(f)
	}
	if f, ok := num(usage, "completion_tokens"); ok {
		res.CompletionTokens = int64(f)
	}
	if d, ok := usage["prompt_tokens_details"].(map[string]any); ok {
		if f, ok := num(d, "cached_tokens"); ok {
			res.CachedTokens = int64(f)
		}
	}
	timings := sub("timings")
	if f, ok := num(timings, "predicted_per_second"); ok {
		res.DecodeTPS = f
	}
	if f, ok := num(timings, "prompt_per_second"); ok {
		res.PrefillTPS = f
	}
	if choices, ok := doc["choices"].([]any); ok && len(choices) > 0 {
		c, _ := choices[0].(map[string]any)
		res.FinishReason, _ = c["finish_reason"].(string)
		if msg, ok := c["message"].(map[string]any); ok {
			res.Content, _ = msg["content"].(string)
			res.ContentBytes = len(res.Content)
		}
	}
	return res, nil
}
