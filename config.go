package imagen

import (
	"os"
	"strconv"
	"time"
)

// Config holds all configuration for imagen providers and storage.
type Config struct {
	OpenAIAPIKey    string
	StabilityAPIKey string
	GCSBucket       string
	GCSObjectPrefix string
	Model           string
	Size            string
	Quality         string
	Provider        ProviderID
	MaxRetries      int
	BaseDelay       time.Duration
	MaxDelay        time.Duration
	HTTPTimeout     time.Duration
	MaxResponseSize int64
	MaxImageSize    int64
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		Model:           "gpt-image-1",
		Size:            "1024x1024",
		Quality:         "standard",
		Provider:        ProviderOpenAI,
		MaxRetries:      5,
		BaseDelay:       time.Second,
		MaxDelay:        30 * time.Second,
		HTTPTimeout:     90 * time.Second,
		MaxResponseSize: 2 << 20,
		MaxImageSize:    25 << 20,
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

// LoadConfigFromEnv reads configuration from environment variables.
// It does not fail on missing optional values; callers should validate
// required fields separately.
func LoadConfigFromEnv() Config {
	cfg := DefaultConfig()

	if v := os.Getenv("OPENAI_API_KEY"); v != "" {
		cfg.OpenAIAPIKey = v
	}
	if v := os.Getenv("STABILITY_API_KEY"); v != "" {
		cfg.StabilityAPIKey = v
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

	return cfg
}
