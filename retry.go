package imagen

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"time"
)

// ShouldRetry is a predicate that returns true if an error is transient
// and the operation should be retried.
type ShouldRetry func(error) bool

// RetryConfig defines retry parameters.
type RetryConfig struct {
	MaxRetries     int
	BaseDelay      time.Duration
	MaxDelay       time.Duration
	ShouldRetry    ShouldRetry
	OperationLabel string
}

// RetryDo executes fn with retries. It returns fn's result on success or the
// last error after exhausting all retries. Context cancellation is
// respected between retries.
func RetryDo[T any](ctx context.Context, cfg RetryConfig, fn func(context.Context) (T, error)) (T, error) {
	base := cfg.BaseDelay
	if base <= 0 {
		base = 200 * time.Millisecond
	}
	maxDelay := cfg.MaxDelay
	if maxDelay <= 0 {
		maxDelay = 30 * time.Second
	}
	maxRetries := cfg.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 3
	}
	shouldRetry := cfg.ShouldRetry
	if shouldRetry == nil {
		shouldRetry = alwaysRetry
	}

	var zero T
	var lastErr error

	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			delay := calcBackoff(base, maxDelay, attempt)
			log.Printf("[retry %s] attempt %d/%d waiting %v", cfg.OperationLabel, attempt+1, maxRetries+1, delay)
			select {
			case <-ctx.Done():
				return zero, ctx.Err()
			case <-time.After(delay):
			}
		}

		result, err := fn(ctx)
		if err == nil {
			if attempt > 0 {
				log.Printf("[retry %s] succeeded on attempt %d/%d", cfg.OperationLabel, attempt+1, maxRetries+1)
			}
			return result, nil
		}

		lastErr = err
		if !shouldRetry(err) {
			return zero, err
		}

		log.Printf("[retry %s] attempt %d/%d failed: %v", cfg.OperationLabel, attempt+1, maxRetries+1, err)
	}

	return zero, fmt.Errorf("%s failed after %d retries: %w", cfg.OperationLabel, maxRetries, lastErr)
}

// RetryDoVoid is like RetryDo but for operations that return only an error.
func RetryDoVoid(ctx context.Context, cfg RetryConfig, fn func(context.Context) error) error {
	_, err := RetryDo[string](ctx, cfg, func(ctx context.Context) (string, error) {
		return "", fn(ctx)
	})
	return err
}

func backoffSleep(ctx context.Context, attempt int, base, maxDelay time.Duration) {
	delay := calcBackoff(base, maxDelay, attempt)
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
	case <-timer.C:
	}
}

func calcBackoff(base, maxDelay time.Duration, attempt int) time.Duration {
	if attempt <= 0 {
		attempt = 1
	}
	exp := base * time.Duration(1<<uint(attempt-1))
	if exp > maxDelay {
		exp = maxDelay
	}
	jitter := time.Duration(float64(exp) * (0.5 + rand.Float64()*0.5))
	return jitter
}

func alwaysRetry(error) bool {
	return true
}
