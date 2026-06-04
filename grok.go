package imagen

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const grokBaseURL = "https://api.x.ai/v1"

type grokImageRequest struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	N              int    `json:"n"`
	AspectRatio    string `json:"aspect_ratio,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"`
}

type grokImageResponse struct {
	Data     []grokImage `json:"data"`
	MIMEType string      `json:"mime_type,omitempty"`
}

type grokImage struct {
	URL     string `json:"url"`
	B64JSON string `json:"b64_json"`
}

type grokErrorResponse struct {
	Error grokError `json:"error"`
}

type grokError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// GrokProvider generates images using the xAI Grok API.
type GrokProvider struct {
	apiKey      string
	baseURL     string
	model       string
	qualityModel string
	client      *http.Client
	maxRespSize int64
}

// GrokOption configures the Grok provider.
type GrokOption func(*GrokProvider)

// WithGrokModel sets the default model for the Grok provider.
func WithGrokModel(model string) GrokOption {
	return func(p *GrokProvider) { p.model = model }
}

// WithGrokQualityModel sets the high-quality model variant.
func WithGrokQualityModel(model string) GrokOption {
	return func(p *GrokProvider) { p.qualityModel = model }
}

// WithGrokBaseURL sets the API base URL.
func WithGrokBaseURL(url string) GrokOption {
	return func(p *GrokProvider) { p.baseURL = url }
}

// WithGrokHTTPClient sets a custom HTTP client.
func WithGrokHTTPClient(c *http.Client) GrokOption {
	return func(p *GrokProvider) { p.client = c }
}

// WithGrokMaxResponseSize sets the maximum response body size in bytes.
func WithGrokMaxResponseSize(n int64) GrokOption {
	return func(p *GrokProvider) { p.maxRespSize = n }
}

// NewGrokProvider creates a new xAI Grok image generation provider.
func NewGrokProvider(apiKey string, opts ...GrokOption) *GrokProvider {
	p := &GrokProvider{
		apiKey:       apiKey,
		baseURL:      grokBaseURL,
		model:        "grok-imagine-image",
		qualityModel: "grok-imagine-image-quality",
		client:       &http.Client{Timeout: 90 * time.Second},
		maxRespSize:  50 << 20,
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// Generate calls the xAI Grok Images API and returns the generated image.
func (p *GrokProvider) Generate(ctx context.Context, req *Request) (*Result, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("grok: %w", ErrAPIKeyRequired)
	}
	if req.Prompt == "" {
		return nil, fmt.Errorf("grok: %w", ErrEmptyPrompt)
	}

	model := p.model
	if req.Quality == "high" || req.Quality == "hd" {
		model = p.qualityModel
	}
	if req.Model != "" {
		model = req.Model
	}

	apiReq := grokImageRequest{
		Model:          model,
		Prompt:         req.Prompt,
		N:              capN(req.N),
		ResponseFormat: "b64_json",
	}

	body, err := json.Marshal(apiReq)
	if err != nil {
		return nil, fmt.Errorf("grok: marshal request: %w", err)
	}

	baseURL := strings.TrimRight(p.baseURL, "/")
	url := baseURL + "/images/generations"

	resp, err := p.doRequest(ctx, url, body)
	if err != nil {
		return nil, fmt.Errorf("grok: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, p.maxRespSize))
	if err != nil {
		return nil, fmt.Errorf("grok: read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var apiErr grokErrorResponse
		if json.Unmarshal(respBody, &apiErr) == nil && apiErr.Error.Message != "" {
			switch apiErr.Error.Code {
			case "rate_limit_exceeded":
				return nil, fmt.Errorf("grok: %w: %s", ErrRateLimited, apiErr.Error.Message)
			}
			return nil, fmt.Errorf("grok: type=%s code=%s msg=%s",
				apiErr.Error.Type, apiErr.Error.Code, apiErr.Error.Message)
		}
		return nil, fmt.Errorf("grok: status=%d body=%s",
			resp.StatusCode, truncate(string(respBody), 500))
	}

	var result grokImageResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		if int64(len(respBody)) >= p.maxRespSize {
			return nil, fmt.Errorf("grok: response truncated at %d bytes, increase max response size", p.maxRespSize)
		}
		return nil, fmt.Errorf("grok: parse response: %w", err)
	}

	if len(result.Data) == 0 {
		return nil, fmt.Errorf("grok: %w", ErrNoImageData)
	}

	img := result.Data[0]
	var imgData []byte

	switch {
	case img.B64JSON != "":
		imgData, err = base64.StdEncoding.DecodeString(img.B64JSON)
		if err != nil {
			return nil, fmt.Errorf("grok: decode base64: %w", err)
		}
	case img.URL != "":
		imgData, err = p.downloadImage(ctx, img.URL)
		if err != nil {
			return nil, fmt.Errorf("grok: download: %w", err)
		}
	default:
		return nil, fmt.Errorf("grok: %w", ErrEmptyURL)
	}

	contentType := result.MIMEType
	if contentType == "" {
		contentType = detectContentType(imgData)
	}

	return &Result{
		Data:        imgData,
		ContentType: contentType,
		Prompt:      req.Prompt,
		Provider:    ProviderGrok,
		CreatedAt:   time.Now(),
	}, nil
}

func (p *GrokProvider) doRequest(ctx context.Context, url string, payload []byte) (*http.Response, error) {
	var resp *http.Response
	var lastErr error

	for attempt := 0; attempt < 5; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if err != nil {
			lastErr = fmt.Errorf("create request: %w", err)
			continue
		}

		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+p.apiKey)

		resp, err = p.client.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("http do: %w", err)
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			if attempt < 4 {
				backoffSleep(ctx, attempt+1, time.Second, 30*time.Second)
			}
			continue
		}

		if shouldRetryHTTP(resp.StatusCode) && attempt < 4 {
			b, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
			resp.Body.Close()
			lastErr = fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
			backoffSleep(ctx, attempt+1, time.Second, 30*time.Second)
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			continue
		}

		return resp, nil
	}

	return nil, fmt.Errorf("request failed after 5 attempts: %w", lastErr)
}

func (p *GrokProvider) downloadImage(ctx context.Context, imageURL string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, imageURL, nil)
	if err != nil {
		return nil, fmt.Errorf("create download request: %w", err)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("image download failed: status=%d", resp.StatusCode)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read image: %w", err)
	}

	return data, nil
}
