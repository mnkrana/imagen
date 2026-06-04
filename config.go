package imagen

import (
	"os"
	"strconv"
	"time"

	"github.com/joho/godotenv"
)

// Config holds all configuration for imagen providers and storage.
type Config struct {
	OpenAIAPIKey         string
	StabilityAPIKey      string
	GrokAPIKey           string
	GCSBucket            string
	GCSObjectPrefix      string
	Model                string
	Size                 string
	Quality              string
	Provider             ProviderID
	MaxRetries           int
	BaseDelay            time.Duration
	MaxDelay             time.Duration
	HTTPTimeout          time.Duration
	MaxResponseSize      int64
	MaxImageSize         int64
	StabilityEndpoint    string
	StabilityModel       string
	StabilityCfgScale    float64
	StabilitySteps       int
	StabilityStylePreset string
	OutputFormat         string
	GrokModel            string
	GrokEndpoint         string
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		Model:                "gpt-image-2",
		Size:                 "1024x1024",
		Quality:              "standard",
		Provider:             ProviderOpenAI,
		MaxRetries:           5,
		BaseDelay:            time.Second,
		MaxDelay:             30 * time.Second,
		HTTPTimeout:          90 * time.Second,
		MaxResponseSize:      2 << 20,
		MaxImageSize:         25 << 20,
		StabilityEndpoint:    "https://api.stability.ai/v1",
		StabilityModel:       "stable-diffusion-xl-1024-v1-0",
		StabilityCfgScale:    7.0,
		StabilitySteps:       30,
		StabilityStylePreset: "",
		OutputFormat:         "webp",
		GrokModel:            "grok-imagine-image",
		GrokEndpoint:         "https://api.x.ai/v1",
	}
}

// Option applies a configuration change.
type Option func(*Config)

// WithOpenAIAPIKey sets the OpenAI API key.
func WithOpenAIAPIKey(key string) Option {
	return func(c *Config) { c.OpenAIAPIKey = key }
}

// WithStabilityAPIKey sets the Stability AI API key.
func WithStabilityAPIKey(key string) Option {
	return func(c *Config) { c.StabilityAPIKey = key }
}

// WithGrokAPIKey sets the xAI Grok API key.
func WithGrokAPIKey(key string) Option {
	return func(c *Config) { c.GrokAPIKey = key }
}

// WithModel sets the image generation model.
func WithModel(model string) Option {
	return func(c *Config) { c.Model = model }
}

// WithSize sets the output image size (e.g. "1024x1024").
func WithSize(size string) Option {
	return func(c *Config) { c.Size = size }
}

// WithQuality sets the image quality ("standard" or "hd").
func WithQuality(q string) Option {
	return func(c *Config) { c.Quality = q }
}

// WithProvider sets the image generation provider.
func WithProvider(p ProviderID) Option {
	return func(c *Config) { c.Provider = p }
}

// WithGCS configures Google Cloud Storage bucket and optional object prefix.
func WithGCS(bucket, prefix string) Option {
	return func(c *Config) {
		c.GCSBucket = bucket
		c.GCSObjectPrefix = prefix
	}
}

// WithRetry configures retry behaviour for provider API calls.
func WithRetry(maxRetries int, baseDelay, maxDelay time.Duration) Option {
	return func(c *Config) {
		c.MaxRetries = maxRetries
		c.BaseDelay = baseDelay
		c.MaxDelay = maxDelay
	}
}

// WithHTTPTimeout sets the HTTP client timeout.
func WithHTTPTimeout(timeout time.Duration) Option {
	return func(c *Config) { c.HTTPTimeout = timeout }
}

// WithSizeLimits sets maximum response body and image download sizes.
func WithSizeLimits(maxResponseSize, maxImageSize int64) Option {
	return func(c *Config) {
		c.MaxResponseSize = maxResponseSize
		c.MaxImageSize = maxImageSize
	}
}

// WithOutputFormat sets the output image format for providers that support it.
func WithOutputFormat(f string) Option {
	return func(c *Config) { c.OutputFormat = f }
}

// LoadConfigFromEnv reads configuration from environment variables.
// It automatically loads .env file from the current directory if present.
// It does not fail on missing optional values; callers should validate
// required fields separately.
func LoadConfigFromEnv() Config {
	if _, err := os.Stat(".env"); err == nil {
		godotenv.Load()
	}

	cfg := DefaultConfig()

	if v := os.Getenv("OPENAI_API_KEY"); v != "" {
		cfg.OpenAIAPIKey = v
	}
	if v := os.Getenv("STABILITY_API_KEY"); v != "" {
		cfg.StabilityAPIKey = v
	}
	if v := os.Getenv("GROK_API_KEY"); v != "" {
		cfg.GrokAPIKey = v
	}
	if v := os.Getenv("GCS_BUCKET"); v != "" {
		cfg.GCSBucket = v
	}
	if v := os.Getenv("GCS_OBJECT_PREFIX"); v != "" {
		cfg.GCSObjectPrefix = v
	}
	if v := os.Getenv("IMAGEN_MODEL"); v != "" {
		cfg.Model = v
	}
	if v := os.Getenv("IMAGEN_SIZE"); v != "" {
		cfg.Size = v
	}
	if v := os.Getenv("IMAGEN_QUALITY"); v != "" {
		cfg.Quality = v
	}
	if v := os.Getenv("IMAGEN_PROVIDER"); v != "" {
		cfg.Provider = ProviderID(v)
	}
	if v := os.Getenv("IMAGEN_MAX_RETRIES"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.MaxRetries = n
		}
	}
	if v := os.Getenv("IMAGEN_HTTP_TIMEOUT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			cfg.HTTPTimeout = d
		}
	}
	if v := os.Getenv("STABILITY_ENDPOINT"); v != "" {
		cfg.StabilityEndpoint = v
	}
	if v := os.Getenv("STABILITY_MODEL"); v != "" {
		cfg.StabilityModel = v
	}
	if v := os.Getenv("STABILITY_CFG_SCALE"); v != "" {
		if n, err := strconv.ParseFloat(v, 64); err == nil && n > 0 {
			cfg.StabilityCfgScale = n
		}
	}
	if v := os.Getenv("STABILITY_STEPS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.StabilitySteps = n
		}
	}
	if v := os.Getenv("STABILITY_STYLE_PRESET"); v != "" {
		cfg.StabilityStylePreset = v
	}
	if v := os.Getenv("IMAGEN_OUTPUT_FORMAT"); v != "" {
		cfg.OutputFormat = v
	}
	if v := os.Getenv("GROK_MODEL"); v != "" {
		cfg.GrokModel = v
	}
	if v := os.Getenv("GROK_ENDPOINT"); v != "" {
		cfg.GrokEndpoint = v
	}

	return cfg
}
