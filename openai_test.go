package imagen

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// tinyPNG returns a minimal valid PNG so tests exercise real image bytes.
func tinyPNG(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 2, 2))); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}

func testProvider(t *testing.T, handler http.HandlerFunc) (*OpenAIProvider, *httptest.Server) {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	p := NewOpenAIProvider("test-key",
		WithOpenAIBaseURL(server.URL),
		WithOpenAIModel("test-model"),
		WithOpenAISize("1024x1536"),
		WithOpenAIQuality("low"),
	)
	return p, server
}

func b64Response(t *testing.T, data []byte) string {
	t.Helper()
	body, err := json.Marshal(openAIResponse{Data: []openAIImage{{B64JSON: base64.StdEncoding.EncodeToString(data)}}})
	if err != nil {
		t.Fatalf("marshal response: %v", err)
	}
	return string(body)
}

func TestOpenAIGenerationsWithoutReferences(t *testing.T) {
	var gotPath string
	p, _ := testProvider(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		if ct := r.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
			t.Errorf("content type = %q, want json", ct)
		}
		var req openAIRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if req.Model != "test-model" || req.Prompt != "a castle" || req.Size != "1024x1536" || req.Quality != "low" {
			t.Fatalf("unexpected request: %+v", req)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(b64Response(t, tinyPNG(t))))
	})

	res, err := p.Generate(context.Background(), &Request{Prompt: "a castle", N: 1})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if gotPath != "/images/generations" {
		t.Fatalf("path = %q, want /images/generations", gotPath)
	}
	if res.ContentType != "image/png" || len(res.Data) == 0 {
		t.Fatalf("unexpected result: type=%s bytes=%d", res.ContentType, len(res.Data))
	}
}

func TestOpenAIEditsWithReferences(t *testing.T) {
	var gotPath string
	p, _ := testProvider(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		if err := r.ParseMultipartForm(10 << 20); err != nil {
			t.Fatalf("parse multipart: %v", err)
		}

		files := r.MultipartForm.File["image[]"]
		if len(files) != 2 {
			t.Fatalf("image[] parts = %d, want 2", len(files))
		}
		for i, fh := range files {
			part, err := fh.Open()
			if err != nil {
				t.Fatalf("open part %d: %v", i, err)
			}
			defer part.Close()
			if got := fh.Header.Get("Content-Type"); got != "image/jpeg" {
				t.Errorf("part %d content type = %q, want image/jpeg", i, got)
			}
		}
		if fh := files[0]; fh.Filename != "selfie.jpg" {
			t.Errorf("filename = %q, want selfie.jpg", fh.Filename)
		}
		if fh := files[1]; fh.Filename != "reference-1.jpg" {
			t.Errorf("generated filename = %q, want reference-1.jpg", fh.Filename)
		}

		if got := r.FormValue("model"); got != "test-model" {
			t.Errorf("model = %q", got)
		}
		if got := r.FormValue("prompt"); got != "a knight" {
			t.Errorf("prompt = %q", got)
		}
		if got := r.FormValue("size"); got != "1024x1536" {
			t.Errorf("size = %q", got)
		}
		if got := r.FormValue("quality"); got != "low" {
			t.Errorf("quality = %q", got)
		}
		if got := r.FormValue("n"); got != "1" {
			t.Errorf("n = %q", got)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(b64Response(t, tinyPNG(t))))
	})

	res, err := p.Generate(context.Background(), &Request{
		Prompt: "a knight",
		ReferenceImages: []ReferenceImage{
			{Data: []byte("fake-jpeg-bytes"), MIMEType: "image/jpeg", Name: "selfie.jpg"},
			{Data: []byte("more-bytes"), MIMEType: "image/jpeg"},
		},
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if gotPath != "/images/edits" {
		t.Fatalf("path = %q, want /images/edits", gotPath)
	}
	if res.ContentType != "image/png" || len(res.Data) == 0 {
		t.Fatalf("unexpected result: type=%s bytes=%d", res.ContentType, len(res.Data))
	}
}

func TestOpenAIEditsRejectsEmptyReference(t *testing.T) {
	p, _ := testProvider(t, func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("API should not be called for an empty reference")
	})
	_, err := p.Generate(context.Background(), &Request{
		Prompt:          "a knight",
		ReferenceImages: []ReferenceImage{{}},
	})
	if err == nil || !strings.Contains(err.Error(), "reference image 0 is empty") {
		t.Fatalf("err = %v, want empty reference error", err)
	}
}

func TestOpenAIErrorMapping(t *testing.T) {
	cases := []struct {
		name    string
		status  int
		body    string
		wantErr error
	}{
		{
			name:    "moderation blocked",
			status:  http.StatusBadRequest,
			body:    `{"error":{"message":"bad","type":"invalid_request_error","code":"moderation_blocked"}}`,
			wantErr: ErrContentFiltered,
		},
		{
			name:    "content policy violation",
			status:  http.StatusBadRequest,
			body:    `{"error":{"message":"bad","type":"invalid_request_error","code":"content_policy_violation"}}`,
			wantErr: ErrContentFiltered,
		},
		{
			name:    "rate limited",
			status:  http.StatusTooManyRequests,
			body:    `{"error":{"message":"slow down","type":"rate_limit_error","code":"rate_limit_exceeded"}}`,
			wantErr: ErrRateLimited,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, _ := testProvider(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			})
			_, err := p.Generate(context.Background(), &Request{Prompt: "x"})
			if err == nil {
				t.Fatal("expected an error")
			}
			if tc.wantErr != nil && !strings.Contains(err.Error(), tc.wantErr.Error()) {
				t.Fatalf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestOpenAIRateLimitOnEditsIsRetryable(t *testing.T) {
	// The edits path must surface ErrRateLimited the same way as generations so
	// the caller's retry loop treats a 429 on either endpoint as transient.
	p, _ := testProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"code":"rate_limit_exceeded","message":"429"}}`))
	})
	_, err := p.Generate(context.Background(), &Request{
		Prompt:          "x",
		ReferenceImages: []ReferenceImage{{Data: []byte("b"), MIMEType: "image/png"}},
	})
	if err == nil || !strings.Contains(err.Error(), ErrRateLimited.Error()) {
		t.Fatalf("err = %v, want %v", err, ErrRateLimited)
	}
}
