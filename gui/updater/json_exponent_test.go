package updater

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestReleaseExponentSizes(t *testing.T) {
	s := httptest.NewTLSServer(http.NotFoundHandler())
	defer s.Close()
	c := testClient(t, s, "")
	m := testManifest(s.URL, make([]byte, 200000))
	m.Assets[0].UnpackedBytes = 64000000000
	m.Runtime.Changed, m.Runtime.DownloadBytes = true, 87790000000
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	for plain, exponent := range map[string]string{`:200000`: `:2e+05`, `:64000000000`: `:6.4e+10`, `:87790000000`: `:8.779e+10`} {
		raw = bytes.ReplaceAll(raw, []byte(plain), []byte(exponent))
	}
	got, asset, failure := c.decodeManifest(raw)
	if failure != nil {
		t.Fatal(failure)
	}
	if asset.Size != 200000 || asset.UnpackedBytes != 64000000000 || got.Runtime.DownloadBytes != 87790000000 {
		t.Fatal("release sizes lost", got, asset)
	}
	if _, _, failure := c.decodeManifest(bytes.ReplaceAll(raw, []byte(`:2e+05`), []byte(`:2.5e+00`))); failure == nil {
		t.Fatal("fractional release size accepted")
	}
}
