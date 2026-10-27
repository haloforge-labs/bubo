package updater

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestUpdateFeedFailuresDistinguishHTTPFromConnectivity(t *testing.T) {
	for _, tc := range []struct {
		status int
		code   Code
	}{
		{http.StatusNotFound, Code("update_information_missing")},
		{http.StatusServiceUnavailable, Code("update_server_error")},
	} {
		t.Run(strconv.Itoa(tc.status), func(t *testing.T) {
			requests := 0
			s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests++
				if !strings.HasSuffix(r.URL.Path, "/bubo-release.json") {
					t.Errorf("check attempted a download: %s", r.URL.Path)
				}
				w.WriteHeader(tc.status)
			}))
			defer s.Close()
			c := testClient(t, s, "")
			result := c.Check(context.Background())
			if result.Status != Failed || result.Failure == nil || result.Failure.Code != tc.code {
				t.Fatalf("HTTP %d check = %+v, failure = %+v; want %s", tc.status, result, result.Failure, tc.code)
			}
			if requests != 1 || result.Failure.URL != c.cfg.ManifestURL || !strings.Contains(result.Failure.Message, "HTTP "+strconv.Itoa(tc.status)) {
				t.Fatalf("missing check evidence: requests=%d failure=%+v", requests, result.Failure)
			}
		})
	}
	t.Run("connection refused", func(t *testing.T) {
		s := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		c := testClient(t, s, "")
		s.Close()
		result := c.Check(context.Background())
		if result.Failure == nil || result.Failure.Code != CodeNetwork {
			t.Fatalf("connection failure = %+v, want network", result.Failure)
		}
	})
	t.Run("connection timeout", func(t *testing.T) {
		s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			<-r.Context().Done()
		}))
		defer s.Close()
		c := testClient(t, s, "")
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()
		result := c.Check(ctx)
		if result.Failure == nil || result.Failure.Code != CodeNetwork {
			t.Fatalf("connection timeout = %+v, want network", result.Failure)
		}
	})
}
