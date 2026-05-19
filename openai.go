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

const openAIBaseURL = "https://api.openai.com/v1/images/generations"

type openAIRequest struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	N              int    `json:"n"`
	Size           string `json:"size,omitempty"`
	Quality        string `json:"quality,omitempty"`
	Style          string `json:"style,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"`
}

type openAIResponse struct {
	Data []openAIImage `json:"data"`
}

type openAIImage struct {
	URL     string `json:"url"`
	B64JSON string `json:"b64_json"`
}

type openAIErrorResponse struct {
	Error openAIError `json:"error"`
}

type openAIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// OpenAIProvider generates images using the OpenAI API.
type OpenAIProvider struct {
	apiKey      string
	model       string
	size        string
	quality     string
	style       string
	client      *http.Client
	respFmt     string
	maxRespSize int64
}

// OpenAIOption configures the OpenAI provider.
type OpenAIOption func(*OpenAIProvider)

// WithOpenAIModel sets the model for the OpenAI provider.
func WithOpenAIModel(model string) OpenAIOption {
	return func(p *OpenAIProvider) { p.model = model }
}

// WithOpenAISize sets the image size.
func WithOpenAISize(size string) OpenAIOption {
	return func(p *OpenAIProvider) { p.size = size }
}

// WithOpenAIQuality sets the image quality ("standard" or "hd").
func WithOpenAIQuality(q string) OpenAIOption {
	return func(p *OpenAIProvider) { p.quality = q }
}

// WithOpenAIStyle sets the image style ("vivid" or "natural").
func WithOpenAIStyle(s string) OpenAIOption {
	return func(p *OpenAIProvider) { p.style = s }
}

// WithOpenAIResponseFormat sets the API response format ("url" or "b64_json").
func WithOpenAIResponseFormat(f string) OpenAIOption {
	return func(p *OpenAIProvider) { p.respFmt = f }
}

// WithOpenAIHTTPClient sets a custom HTTP client.
func WithOpenAIHTTPClient(c *http.Client) OpenAIOption {
	return func(p *OpenAIProvider) { p.client = c }
}

// WithOpenAIMaxResponseSize sets the maximum response body size in bytes.
func WithOpenAIMaxResponseSize(n int64) OpenAIOption {
	return func(p *OpenAIProvider) { p.maxRespSize = n }
}

// NewOpenAIProvider creates a new OpenAI image generation provider.
func NewOpenAIProvider(apiKey string, opts ...OpenAIOption) *OpenAIProvider {
	p := &OpenAIProvider{
		apiKey:      apiKey,
		model:       "gpt-image-2",
		size:        "1024x1024",
		quality:     "standard",
		style:       "",
		client:      &http.Client{Timeout: 90 * time.Second},
		respFmt:     "",
		maxRespSize: 50 << 20,
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// Generate calls the OpenAI Images API and returns the generated image.
func (p *OpenAIProvider) Generate(ctx context.Context, req *Request) (*Result, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("openai: %w", ErrAPIKeyRequired)
	}
	if req.Prompt == "" {
		return nil, fmt.Errorf("openai: %w", ErrEmptyPrompt)
	}

	apiReq := openAIRequest{
		Model:          coalesce(req.Model, p.model),
		Prompt:         req.Prompt,
		N:              capN(req.N),
		Size:           coalesce(req.Size, p.size),
		Quality:        coalesce(req.Quality, p.quality),
		Style:          coalesce(req.Style, p.style),
		ResponseFormat: p.respFmt,
	}

	body, err := json.Marshal(apiReq)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	resp, err := p.doRequest(ctx, openAIBaseURL, body)
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, p.maxRespSize))
	if err != nil {
		return nil, fmt.Errorf("openai: read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var apiErr openAIErrorResponse
		if json.Unmarshal(respBody, &apiErr) == nil && apiErr.Error.Message != "" {
			switch apiErr.Error.Code {
			case "content_policy_violation":
				return nil, fmt.Errorf("openai: %w: %s", ErrContentFiltered, apiErr.Error.Message)
			case "rate_limit_exceeded":
				return nil, fmt.Errorf("openai: %w: %s", ErrRateLimited, apiErr.Error.Message)
			}
			return nil, fmt.Errorf("openai: type=%s code=%s msg=%s",
				apiErr.Error.Type, apiErr.Error.Code, apiErr.Error.Message)
		}
		return nil, fmt.Errorf("openai: status=%d body=%s",
			resp.StatusCode, truncate(string(respBody), 500))
	}

	var result openAIResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		if int64(len(respBody)) >= p.maxRespSize {
			return nil, fmt.Errorf("openai: response truncated at %d bytes, increase max response size", p.maxRespSize)
		}
		return nil, fmt.Errorf("openai: parse response: %w", err)
	}

	if len(result.Data) == 0 {
		return nil, fmt.Errorf("openai: %w", ErrNoImageData)
	}

	img := result.Data[0]
	var imgData []byte

	switch {
	case img.B64JSON != "":
		imgData, err = base64.StdEncoding.DecodeString(img.B64JSON)
		if err != nil {
			return nil, fmt.Errorf("openai: decode base64: %w", err)
		}
	case img.URL != "":
		imgData, err = p.downloadImage(ctx, img.URL)
		if err != nil {
			return nil, fmt.Errorf("openai: download: %w", err)
		}
	default:
		return nil, fmt.Errorf("openai: %w", ErrEmptyURL)
	}

	return &Result{
		Data:        imgData,
		ContentType: detectContentType(imgData),
		Prompt:      req.Prompt,
		Provider:    ProviderOpenAI,
		CreatedAt:   time.Now(),
	}, nil
}

func (p *OpenAIProvider) doRequest(ctx context.Context, url string, payload []byte) (*http.Response, error) {
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

func (p *OpenAIProvider) downloadImage(ctx context.Context, imageURL string) ([]byte, error) {
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

func shouldRetryHTTP(statusCode int) bool {
	switch statusCode {
	case http.StatusTooManyRequests:
		return true
	default:
		return statusCode >= 500
	}
}

func detectContentType(data []byte) string {
	if len(data) < 12 {
		return "image/png"
	}
	switch {
	case len(data) > 8 && string(data[:8]) == "\x89PNG\r\n\x1a\n":
		return "image/png"
	case len(data) > 2 && data[0] == 0xFF && data[1] == 0xD8:
		return "image/jpeg"
	case len(data) > 4 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP":
		return "image/webp"
	default:
		return "image/png"
	}
}

func capN(n int) int {
	if n <= 0 {
		return 1
	}
	if n > 10 {
		return 10
	}
	return n
}

func coalesce(s ...string) string {
	for _, v := range s {
		if v != "" {
			return v
		}
	}
	return ""
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}
