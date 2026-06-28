# imagen

> **Unified Go library for multi-provider AI image generation (OpenAI, Stability AI, xAI Grok) with automatic cloud storage upload.**

[![Go Version](https://img.shields.io/badge/Go-1.26+-00ADD8?logo=go)](https://golang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Go Reference](https://img.shields.io/badge/Reference-pkg.go.dev-007D9C?logo=go)](https://pkg.go.dev/github.com/mnkrana/imagen)

Stop copy-pasting the same OpenAI + GCS pipeline across your Go projects. **imagen** provides a clean `Provider` / `Storage` interface so you can swap AI providers, change storage backends, and generate-and-store images in a single call.

---

## Quick Start

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/mnkrana/imagen"
)

func main() {
	cfg := imagen.LoadConfigFromEnv()

	client, err := imagen.NewClientFromConfig(cfg)
	if err != nil {
		log.Fatalf("client: %v", err)
	}

	result, err := client.GenerateAndStore(context.Background(), &imagen.Request{
		Prompt: "a serene mountain landscape at sunset, photorealistic",
	})
	if err != nil {
		log.Fatalf("generate: %v", err)
	}

	fmt.Println("Generated image:")
	fmt.Printf("  URL:  %s\n", result.URL)
	fmt.Printf("  Size: %d bytes\n", result.Size)
}
```

**Run it:**

```bash
cp .env.example .env      # fill in your keys and bucket
go run .
```

---

## Installation

```bash
go get github.com/mnkrana/imagen
```

Requires Go 1.26+.

---

## Configuration

### Environment variables

imagen auto-loads a `.env` file from the current directory if present. Set these variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | for OpenAI | — | OpenAI API key |
| `STABILITY_API_KEY` | for Stability | — | Stability AI API key |
| `GCS_BUCKET` | yes | — | Google Cloud Storage bucket name |
| `IMAGEN_PROVIDER` | no | `openai` | Provider: `openai`, `stability`, or `grok` |
| `IMAGEN_MODEL` | no | `gpt-image-2` | Model name |
| `IMAGEN_SIZE` | no | `1024x1024` | Output size |
| `IMAGEN_QUALITY` | no | `standard` | Quality: `standard` or `hd` |
| `IMAGEN_OUTPUT_FORMAT` | no | `webp` | Output format (Stability: `png`, `webp`, `jpeg`) |
| `IMAGEN_MAX_RETRIES` | no | `5` | Max retry attempts for API calls |
| `IMAGEN_HTTP_TIMEOUT` | no | `90s` | HTTP client timeout |
| `GCS_OBJECT_PREFIX` | no | `images/` | GCS object path prefix |
| `STABILITY_ENDPOINT` | no | `https://api.stability.ai/v1` | Stability API base URL |
| `STABILITY_MODEL` | no | `stable-diffusion-xl-1024-v1-0` | Stability model ID |
| `STABILITY_CFG_SCALE` | no | `7.0` | Classifier-free guidance scale |
| `STABILITY_STEPS` | no | `30` | Number of inference steps |
| `STABILITY_STYLE_PRESET` | no | — | Style preset (`photographic`, `digital-art`, `anime`, etc.) |
| `GROK_API_KEY` | for Grok | — | xAI Grok API key |
| `GROK_MODEL` | no | `grok-imagine-image` | Grok model name |
| `GROK_ENDPOINT` | no | `https://api.x.ai/v1` | Grok API base URL |

### Programmatic configuration

```go
provider := imagen.NewOpenAIProvider("sk-...",
    imagen.WithOpenAIModel("gpt-image-2"),
    imagen.WithOpenAISize("1792x1024"),
    imagen.WithOpenAIQuality("hd"),
)

storage := imagen.NewGCSStorage("my-bucket",
    imagen.WithGCSObjectNamer(func(r *imagen.Result) string {
        return fmt.Sprintf("generated/%d%s", time.Now().UnixNano(), ".png")
    }),
)

client := imagen.NewClient(provider, storage)
```

Or with Grok:

```go
provider := imagen.NewGrokProvider("xai-...",
    imagen.WithGrokQualityModel("grok-imagine-image-quality"),
)

client := imagen.NewClient(provider, storage,
    imagen.WithRateLimit(rate.Every(12*time.Second), 5),
)
```

---

## Architecture

```
┌──────────┐    ┌──────────────────┐    ┌─────────────┐    ┌──────────┐
│  Request  │───▶│  Client          │───▶│  Provider   │───▶│  Result  │
│  (prompt) │    │  GenerateAndStore│    │  (OpenAI /  │    │  (bytes) │
└──────────┘    │                  │    │   Stability/│    └──────────┘
                │                  │    └─────────────┘         │
                │                  │                            ▼
                │                  │    ┌─────────────┐    ┌──────────┐
                │                  │───▶│  Storage    │───▶│Storage   │
                │                  │    │  (GCS)      │    │Result    │
                └──────────────────┘    └─────────────┘    │(URL)     │
                                                            └──────────┘
```

Two core interfaces:

```go
type Provider interface {
	Generate(ctx context.Context, req *Request) (*Result, error)
}

type Storage interface {
	Upload(ctx context.Context, result *Result) (*StorageResult, error)
}
```

`Client.GenerateAndStore` calls them in sequence, wrapping errors at each step.

---

## Providers

### OpenAI

```go
provider := imagen.NewOpenAIProvider("sk-...")
```

Supports `gpt-image-2`, `dall-e-3`, and `dall-e-2`. Configurable via options:

- `WithOpenAIModel` — model name
- `WithOpenAISize` — output dimensions
- `WithOpenAIQuality` — `standard` or `hd`
- `WithOpenAIStyle` — `vivid` or `natural`
- `WithOpenAIResponseFormat` — `url` or `b64_json`
- `WithOpenAIHTTPClient` — custom HTTP client

Handles `b64_json` and `url` response formats automatically, detects content type via magic bytes, and maps API errors to typed sentinels (`content_policy_violation`, `moderation_blocked`, `rate_limit_exceeded`, etc.).

**Content-filter sanitization:** When a prompt triggers a content policy violation (`content_policy_violation` or `moderation_blocked`), the provider automatically retries with progressively safer prompts:

1. **Tier 1** — Prepends `"Fictional storybook illustration from a literary narrative. "` to the original prompt.
2. **Tier 2** — If Tier 1 is also blocked and `Request.SafeFallback` is set, the fallback prompt is used instead.

```go
result, err := client.GenerateAndStore(ctx, &imagen.Request{
    Prompt:       "war scene with explosions",
    SafeFallback: "a peaceful landscape at sunset",
})
```

This prevents hard failures from over-aggressive moderation while giving the caller full control over the fallback content.

### Stability AI

```go
provider := imagen.NewStabilityProvider("...")
```

Supports SDXL, SD3.5, and any Stability AI text-to-image model. Configurable via options:

- `WithStabilityEndpoint` — API base URL
- `WithStabilityModel` — model ID (e.g. `stable-diffusion-xl-1024-v1-0`)
- `WithStabilityCfgScale` — classifier-free guidance scale
- `WithStabilitySteps` — inference steps
- `WithStabilityStylePreset` — style preset (`photographic`, `digital-art`, `anime`, etc.)
- `WithStabilityOutputFormat` — output format (`png`, `webp`, `jpeg`)
- `WithStabilityHTTPClient` — custom HTTP client

Size is parsed from `Request.Size` (e.g. `"1024x1024"`). Returns the seed from the first artifact and detects content type from magic bytes.

### xAI Grok

```go
provider := imagen.NewGrokProvider("xai-...")
```

Supports `grok-imagine-image` and `grok-imagine-image-quality` models. Selects the quality model automatically when `Request.Quality` is `"high"` or `"hd"`. Configurable via options:

- `WithGrokModel` — default model name (default: `grok-imagine-image`)
- `WithGrokQualityModel` — high-quality model variant (default: `grok-imagine-image-quality`)
- `WithGrokBaseURL` — API base URL
- `WithGrokHTTPClient` — custom HTTP client

Handles both `b64_json` and `url` response formats, respects `mime_type` from the API response, and detects content type from magic bytes as fallback.

---

## Storage

### Google Cloud Storage

```go
storage := imagen.NewGCSStorage("my-bucket")
```

Creates Firebase Storage-compatible download URLs by default. Customizable via options:

- `WithGCSObjectNamer` — custom path generation function
- `WithGCSPublicURL` — custom URL builder

Also provides:
- `UploadReader` — for streaming image data from a reader
- `EnsureFirebaseToken` — retrieve or create Firebase download tokens for existing objects

**Authentication:** Uses [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials). For local dev:

```bash
gcloud auth application-default login
```

For production (Cloud Run, GCE, GKE), the default service account is used automatically.

---

## Errors

All provider and storage errors are wrapped with `fmt.Errorf("...: %w", err)`, making them inspectable with `errors.Is`. Sentinel errors:

```go
var (
	ErrAPIKeyRequired                  // API key is required
	ErrGCSBucketRequired               // GCS bucket is required
	ErrNoImageData                     // no image data returned
	ErrEmptyURL                        // empty image URL
	ErrContentFiltered                 // content filtered by provider policy
	ErrRateLimited                     // rate limited by provider
	ErrUploadFailed                    // upload to storage failed
	ErrInvalidSize                     // invalid image size format
	ErrEmptyPrompt                     // prompt is required
)
```

Example:

```go
result, err := client.GenerateAndStore(ctx, req)
if errors.Is(err, imagen.ErrContentFiltered) {
    log.Println("Prompt was filtered by safety policy")
}
```

---

## Retry Utility

The built-in generic retry utility is available for any operation:

```go
result, err := imagen.RetryDo(ctx, imagen.RetryConfig{
    MaxRetries:     3,
    BaseDelay:      1 * time.Second,
    MaxDelay:       30 * time.Second,
    OperationLabel: "my_operation",
}, func(ctx context.Context) (string, error) {
    return myFlakyOperation(ctx)
})
```

Features exponential backoff with jitter, context cancellation, and configurable `ShouldRetry` predicate.

---

## Examples

Full working examples are in the [`examples/`](examples/) directory:

- [`basic`](examples/basic/) — minimal end-to-end generation and upload

Run any example with:

```bash
cd examples/basic
cp .env.example .env   # fill in your credentials
go run .
```

---

## Project Structure

```
imagen/
├── imagen.go       — Core types, interfaces, Client orchestrator
├── openai.go       — OpenAI provider implementation
├── stability.go    — Stability AI provider implementation
├── grok.go         — xAI Grok provider implementation
├── gcs.go          — GCS storage implementation
├── config.go       — Config, options, env loading
├── errors.go       — Sentinel errors
├── retry.go        — Generic retry utility
├── retry_test.go   — Retry tests
├── examples/       — Runnable examples
├── Makefile        — Build, test, lint targets
└── .env.example    — Environment variable template
```

---

## Development

```bash
make test         # run tests
make test-race    # run with race detector
make lint         # run golangci-lint
make build        # compile all packages
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

[Mayank Rana](https://github.com/mnkrana) — built for reusable, production-grade AI image pipelines.
