package imagen

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"strconv"
	"strings"
	"time"
)

const openAIDefaultBaseURL = "https://api.openai.com/v1"

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
	baseURL     string
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

// WithOpenAIBaseURL overrides the API base URL. Defaults to
// https://api.openai.com/v1. Useful for tests and proxies.
func WithOpenAIBaseURL(url string) OpenAIOption {
	return func(p *OpenAIProvider) { p.baseURL = url }
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
		baseURL:     openAIDefaultBaseURL,
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
// When the request carries reference images, the edits endpoint is used so the
// output is conditioned on them; otherwise the plain generations endpoint is
// used. If the prompt triggers a content policy violation, it retries with
// progressively sanitized prompts (Tier 1: safety prefix, Tier 2: SafeFallback).
func (p *OpenAIProvider) Generate(ctx context.Context, req *Request) (*Result, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("openai: %w", ErrAPIKeyRequired)
	}
	if req.Prompt == "" {
		return nil, fmt.Errorf("openai: %w", ErrEmptyPrompt)
	}
	if len(req.ReferenceImages) > 0 {
		for i, ref := range req.ReferenceImages {
			if len(ref.Data) == 0 {
				return nil, fmt.Errorf("openai: reference image %d is empty", i)
			}
		}
	}

	// Attempt 1: original prompt
	result, err := p.callAPI(ctx, req.Prompt, req)
	if err == nil {
		return result, nil
	}

	if !errors.Is(err, ErrContentFiltered) {
		return nil, err
	}

	// ── Tier 1: prepend safety prefix ──
	log.Printf("[SANITIZE] OpenAI moderation blocked original prompt")
	log.Printf("[SANITIZE] Original: %.300s", req.Prompt)

	safePrompt := "Fictional storybook illustration from a literary narrative. " + req.Prompt
	log.Printf("[SANITIZE] Tier 1 retry: %.300s", safePrompt)

	result, err = p.callAPI(ctx, safePrompt, req)
	if err == nil {
		log.Printf("[SANITIZE] Tier 1 retry succeeded")
		return result, nil
	}

	if !errors.Is(err, ErrContentFiltered) {
		return nil, err
	}

	// ── Tier 2: use SafeFallback from caller ──
	if req.SafeFallback != "" {
		log.Printf("[SANITIZE] Tier 1 failed, using SafeFallback")
		log.Printf("[SANITIZE] Fallback: %.300s", req.SafeFallback)

		result, err = p.callAPI(ctx, req.SafeFallback, req)
		if err == nil {
			log.Printf("[SANITIZE] Tier 2 retry succeeded")
			return result, nil
		}
		if errors.Is(err, ErrContentFiltered) {
			log.Printf("[SANITIZE] Tier 2 also blocked: %v", err)
		} else {
			log.Printf("[SANITIZE] Tier 2 failed: %v", err)
		}
	}

	return nil, err
}

// callAPI performs a single API call for the given prompt, choosing the edits
// endpoint when the request carries reference images and the generations
// endpoint otherwise.
func (p *OpenAIProvider) callAPI(ctx context.Context, prompt string, req *Request) (*Result, error) {
	if len(req.ReferenceImages) > 0 {
		return p.callEditAPI(ctx, prompt, req)
	}
	return p.callGenerateAPI(ctx, prompt, req)
}

// callGenerateAPI hits the text-to-image generations endpoint.
func (p *OpenAIProvider) callGenerateAPI(ctx context.Context, prompt string, req *Request) (*Result, error) {
	apiReq := openAIRequest{
		Model:          coalesce(req.Model, p.model),
		Prompt:         prompt,
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

	resp, err := p.doRequest(ctx, p.baseURL+"/images/generations", body, "application/json")
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, p.maxRespSize))
	if err != nil {
		return nil, fmt.Errorf("openai: read response: %w", err)
	}
	if err := p.checkStatus(resp, respBody); err != nil {
		return nil, err
	}
	return p.decodeResponse(ctx, respBody, prompt, nil)
}

// callEditAPI hits the image edits endpoint, sending each reference image as an
// image[] multipart part so the output is conditioned on it.
func (p *OpenAIProvider) callEditAPI(ctx context.Context, prompt string, req *Request) (*Result, error) {
	body, contentType, err := p.buildEditBody(prompt, req)
	if err != nil {
		return nil, fmt.Errorf("openai: build edit request: %w", err)
	}

	resp, err := p.doRequest(ctx, p.baseURL+"/images/edits", body, contentType)
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, p.maxRespSize))
	if err != nil {
		return nil, fmt.Errorf("openai: read response: %w", err)
	}
	if err := p.checkStatus(resp, respBody); err != nil {
		return nil, err
	}
	return p.decodeResponse(ctx, respBody, prompt, nil)
}

// checkStatus maps a non-200 response to the package error taxonomy. Both
// image endpoints return the same error JSON shape.
func (p *OpenAIProvider) checkStatus(resp *http.Response, respBody []byte) error {
	if resp.StatusCode == http.StatusOK {
		return nil
	}
	var apiErr openAIErrorResponse
	if json.Unmarshal(respBody, &apiErr) == nil && apiErr.Error.Message != "" {
		switch apiErr.Error.Code {
		case "content_policy_violation", "moderation_blocked":
			return fmt.Errorf("openai: %w: %s", ErrContentFiltered, apiErr.Error.Message)
		case "rate_limit_exceeded":
			return fmt.Errorf("openai: %w: %s", ErrRateLimited, apiErr.Error.Message)
		}
		return fmt.Errorf("openai: type=%s code=%s msg=%s",
			apiErr.Error.Type, apiErr.Error.Code, apiErr.Error.Message)
	}
	return fmt.Errorf("openai: status=%d body=%s",
		resp.StatusCode, truncate(string(respBody), 500))
}

// buildEditBody renders the multipart body for the edits endpoint. Every
// reference image is attached as an image[] part carrying its own MIME type.
func (p *OpenAIProvider) buildEditBody(prompt string, req *Request) ([]byte, string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	for i, ref := range req.ReferenceImages {
		name := ref.Name
		if name == "" {
			name = fmt.Sprintf("reference-%d%s", i, referenceExt(ref.MIMEType))
		}
		mimeType := ref.MIMEType
		if mimeType == "" {
			mimeType = "image/png"
		}
		h := textproto.MIMEHeader{}
		h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="image[]"; filename=%q`, name))
		h.Set("Content-Type", mimeType)
		part, err := w.CreatePart(h)
		if err != nil {
			return nil, "", err
		}
		if _, err := part.Write(ref.Data); err != nil {
			return nil, "", err
		}
	}

	if err := w.WriteField("model", coalesce(req.Model, p.model)); err != nil {
		return nil, "", err
	}
	if err := w.WriteField("prompt", prompt); err != nil {
		return nil, "", err
	}
	if err := w.WriteField("n", strconv.Itoa(capN(req.N))); err != nil {
		return nil, "", err
	}
	if size := coalesce(req.Size, p.size); size != "" {
		if err := w.WriteField("size", size); err != nil {
			return nil, "", err
		}
	}
	if quality := coalesce(req.Quality, p.quality); quality != "" {
		if err := w.WriteField("quality", quality); err != nil {
			return nil, "", err
		}
	}
	if err := w.Close(); err != nil {
		return nil, "", err
	}
	return buf.Bytes(), w.FormDataContentType(), nil
}

// referenceExt maps a MIME type to a filename extension for multipart parts.
func referenceExt(mimeType string) string {
	switch mimeType {
	case "image/jpeg":
		return ".jpg"
	case "image/webp":
		return ".webp"
	default:
		return ".png"
	}
}

// decodeResponse parses the shared response shape of the generations and edits
// endpoints and downloads or decodes the first image.
func (p *OpenAIProvider) decodeResponse(ctx context.Context, respBody []byte, prompt string, prevErr error) (*Result, error) {
	if prevErr != nil {
		return nil, prevErr
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
	var err error

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
		Prompt:      prompt,
		Provider:    ProviderOpenAI,
		CreatedAt:   time.Now(),
	}, nil
}

func (p *OpenAIProvider) doRequest(ctx context.Context, url string, payload []byte, contentType string) (*http.Response, error) {
	var resp *http.Response
	var lastErr error

	for attempt := 0; attempt < 5; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if err != nil {
			lastErr = fmt.Errorf("create request: %w", err)
			continue
		}

		req.Header.Set("Content-Type", contentType)
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
