package imagen

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

type stabilityRequest struct {
	TextPrompts  []stabilityTextPrompt `json:"text_prompts"`
	CfgScale     float64               `json:"cfg_scale,omitempty"`
	Steps        int                   `json:"steps,omitempty"`
	StylePreset  string                `json:"style_preset,omitempty"`
	Width        int                   `json:"width"`
	Height       int                   `json:"height"`
	OutputFormat string                `json:"output_format,omitempty"`
}

type stabilityTextPrompt struct {
	Text   string  `json:"text"`
	Weight float64 `json:"weight,omitempty"`
}

type stabilityResponse struct {
	Artifacts []stabilityArtifact `json:"artifacts"`
}

type stabilityArtifact struct {
	Base64       string `json:"base64"`
	Seed         int    `json:"seed"`
	FinishReason string `json:"finishReason"`
}

type stabilityErrorResponse struct {
	Message string   `json:"message"`
	Errors  []string `json:"errors"`
}

// StabilityProvider generates images using the Stability AI API.
type StabilityProvider struct {
	apiKey      string
	endpoint    string
	model       string
	cfgScale    float64
	steps       int
	stylePreset string
	outputFmt   string
	client      *http.Client
	maxRespSize int64
}

// StabilityOption configures the Stability AI provider.
type StabilityOption func(*StabilityProvider)

// WithStabilityEndpoint sets the API base endpoint.
func WithStabilityEndpoint(endpoint string) StabilityOption {
	return func(p *StabilityProvider) { p.endpoint = endpoint }
}

// WithStabilityModel sets the Stability AI model.
func WithStabilityModel(model string) StabilityOption {
	return func(p *StabilityProvider) { p.model = model }
}

// WithStabilityCfgScale sets the classifier-free guidance scale.
func WithStabilityCfgScale(scale float64) StabilityOption {
	return func(p *StabilityProvider) { p.cfgScale = scale }
}

// WithStabilitySteps sets the number of inference steps.
func WithStabilitySteps(steps int) StabilityOption {
	return func(p *StabilityProvider) { p.steps = steps }
}

// WithStabilityStylePreset sets the style preset (e.g. "photographic", "digital-art").
func WithStabilityStylePreset(preset string) StabilityOption {
	return func(p *StabilityProvider) { p.stylePreset = preset }
}

// WithStabilityOutputFormat sets the output image format ("png", "webp", "jpeg").
func WithStabilityOutputFormat(f string) StabilityOption {
	return func(p *StabilityProvider) { p.outputFmt = f }
}

// WithStabilityHTTPClient sets a custom HTTP client.
func WithStabilityHTTPClient(c *http.Client) StabilityOption {
	return func(p *StabilityProvider) { p.client = c }
}

// WithStabilityMaxResponseSize sets the maximum response body size in bytes.
func WithStabilityMaxResponseSize(n int64) StabilityOption {
	return func(p *StabilityProvider) { p.maxRespSize = n }
}

// NewStabilityProvider creates a new Stability AI image generation provider.
func NewStabilityProvider(apiKey string, opts ...StabilityOption) *StabilityProvider {
	p := &StabilityProvider{
		apiKey:      apiKey,
		endpoint:    "https://api.stability.ai/v1",
		model:       "stable-diffusion-xl-1024-v1-0",
		cfgScale:    7.0,
		steps:       30,
		stylePreset: "",
		outputFmt:   "webp",
		client:      &http.Client{Timeout: 90 * time.Second},
		maxRespSize: 50 << 20,
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// Generate calls the Stability AI API and returns the generated image.
func (p *StabilityProvider) Generate(ctx context.Context, req *Request) (*Result, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("stability: %w", ErrAPIKeyRequired)
	}
	if req.Prompt == "" {
		return nil, fmt.Errorf("stability: %w", ErrEmptyPrompt)
	}

	size := coalesce(req.Size, "1024x1024")
	width, height, err := parseSize(size)
	if err != nil {
		return nil, fmt.Errorf("stability: %w", err)
	}

	apiReq := stabilityRequest{
		TextPrompts: []stabilityTextPrompt{
			{Text: req.Prompt, Weight: 1.0},
		},
		CfgScale:     p.cfgScale,
		Steps:        p.steps,
		Width:        width,
		Height:       height,
		OutputFormat: p.outputFmt,
	}
	if p.stylePreset != "" {
		apiReq.StylePreset = p.stylePreset
	}

	body, err := json.Marshal(apiReq)
	if err != nil {
		return nil, fmt.Errorf("stability: marshal request: %w", err)
	}

	endpoint := strings.TrimRight(p.endpoint, "/")
	url := fmt.Sprintf("%s/generation/%s/text-to-image", endpoint, p.model)

	httpResp, err := p.doRequest(ctx, url, body)
	if err != nil {
		return nil, fmt.Errorf("stability: %w", err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(httpResp.Body, p.maxRespSize))
	if err != nil {
		return nil, fmt.Errorf("stability: read response: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		var apiErr stabilityErrorResponse
		if json.Unmarshal(respBody, &apiErr) == nil && apiErr.Message != "" {
			return nil, fmt.Errorf("stability: status=%d msg=%s", httpResp.StatusCode, apiErr.Message)
		}
		return nil, fmt.Errorf("stability: status=%d body=%s",
			httpResp.StatusCode, truncate(string(respBody), 500))
	}

	var result stabilityResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		if int64(len(respBody)) >= p.maxRespSize {
			return nil, fmt.Errorf("stability: response truncated at %d bytes, increase max response size", p.maxRespSize)
		}
		return nil, fmt.Errorf("stability: parse response: %w", err)
	}

	if len(result.Artifacts) == 0 {
		return nil, fmt.Errorf("stability: %w", ErrNoImageData)
	}

	art := result.Artifacts[0]
	imgData, err := base64.StdEncoding.DecodeString(art.Base64)
	if err != nil {
		return nil, fmt.Errorf("stability: decode base64: %w", err)
	}

	return &Result{
		Data:        imgData,
		ContentType: detectContentType(imgData),
		Seed:        art.Seed,
		Prompt:      req.Prompt,
		Provider:    ProviderStability,
		CreatedAt:   time.Now(),
	}, nil
}

func (p *StabilityProvider) doRequest(ctx context.Context, url string, payload []byte) (*http.Response, error) {
	var resp *http.Response
	var lastErr error

	for attempt := 0; attempt < 5; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if err != nil {
			lastErr = fmt.Errorf("create request: %w", err)
			continue
		}

		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json")
		req.Header.Set("Authorization", "Bearer " + p.apiKey)

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

func parseSize(size string) (width, height int, err error) {
	parts := strings.SplitN(size, "x", 2)
	if len(parts) != 2 {
		return 0, 0, fmt.Errorf("invalid size %q: expected WxH format", size)
	}
	width, err = strconv.Atoi(parts[0])
	if err != nil {
		return 0, 0, fmt.Errorf("parse width from %q: %w", size, err)
	}
	height, err = strconv.Atoi(parts[1])
	if err != nil {
		return 0, 0, fmt.Errorf("parse height from %q: %w", size, err)
	}
	return width, height, nil
}
