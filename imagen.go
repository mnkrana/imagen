// Package imagen provides a unified interface for AI image generation
// across multiple providers (OpenAI, Stability AI) with automatic
// upload to cloud storage (Google Cloud Storage).
package imagen

import (
	"context"
	"fmt"
	"time"
)

// ProviderID identifies an image generation provider.
type ProviderID string

const (
	ProviderOpenAI    ProviderID = "openai"
	ProviderStability ProviderID = "stability"
)

// Request defines parameters for generating an image.
type Request struct {
	Prompt  string
	Model   string
	Size    string
	Quality string
	N       int
	Style   string

	Extras map[string]any
}

// Result holds the generated image data and metadata.
type Result struct {
	Data        []byte
	ContentType string
	Seed        int
	Prompt      string
	Provider    ProviderID
	CreatedAt   time.Time
}

// StorageResult holds the result of persisting an image.
type StorageResult struct {
	URL         string
	Bucket      string
	ObjectPath  string
	ContentType string
	Size        int64
	CreatedAt   time.Time
}

// Provider generates images from text prompts.
type Provider interface {
	Generate(ctx context.Context, req *Request) (*Result, error)
}

// Storage persists generated images and returns a publicly accessible URL.
type Storage interface {
	Upload(ctx context.Context, result *Result) (*StorageResult, error)
}

// Client orchestrates image generation and storage.
type Client struct {
	provider Provider
	storage  Storage
}

// NewClient creates a new Client with the given provider and storage backend.
func NewClient(provider Provider, storage Storage) *Client {
	return &Client{
		provider: provider,
		storage:  storage,
	}
}

// GenerateAndStore generates an image and uploads it to storage.
// It returns the storage result containing the public URL.
func (c *Client) GenerateAndStore(ctx context.Context, req *Request) (*StorageResult, error) {
	result, err := c.provider.Generate(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("generate: %w", err)
	}

	storageResult, err := c.storage.Upload(ctx, result)
	if err != nil {
		return nil, fmt.Errorf("store: %w", err)
	}

	return storageResult, nil
}

// Provider returns the underlying provider.
func (c *Client) Provider() Provider {
	return c.provider
}

// Storage returns the underlying storage backend.
func (c *Client) Storage() Storage {
	return c.storage
}

// NewClientFromConfig creates a fully configured Client from a Config.
// It selects the provider based on cfg.Provider and sets up a GCS storage backend.
func NewClientFromConfig(cfg Config) (*Client, error) {
	if cfg.GCSBucket == "" {
		return nil, ErrGCSBucketRequired
	}

	var provider Provider
	switch cfg.Provider {
	case ProviderOpenAI:
		if cfg.OpenAIAPIKey == "" {
			return nil, ErrAPIKeyRequired
		}
		provider = NewOpenAIProvider(cfg.OpenAIAPIKey,
			WithOpenAIModel(cfg.Model),
			WithOpenAISize(cfg.Size),
			WithOpenAIQuality(cfg.Quality),
		)
	case ProviderStability:
		if cfg.StabilityAPIKey == "" {
			return nil, ErrAPIKeyRequired
		}
		return nil, fmt.Errorf("stability provider not yet implemented")
	default:
		return nil, fmt.Errorf("unknown provider: %s", cfg.Provider)
	}

	storage := NewGCSStorage(cfg.GCSBucket)

	return NewClient(provider, storage), nil
}
