package imagen

import "errors"

var (
	ErrAPIKeyRequired    = errors.New("API key is required")
	ErrGCSBucketRequired = errors.New("GCS bucket is required")
	ErrNoImageData       = errors.New("no image data returned")
	ErrEmptyURL          = errors.New("empty image URL")
	ErrContentFiltered   = errors.New("content filtered by provider policy")
	ErrRateLimited       = errors.New("rate limited by provider")
	ErrUploadFailed      = errors.New("upload to storage failed")
	ErrInvalidSize       = errors.New("invalid image size format, expected e.g. 1024x1024")
	ErrEmptyPrompt       = errors.New("prompt is required")
)
