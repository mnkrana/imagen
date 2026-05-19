package imagen

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func TestRetryDoSuccessFirstAttempt(t *testing.T) {
	ctx := context.Background()
	result, err := RetryDo(ctx, RetryConfig{
		MaxRetries:     3,
		BaseDelay:      1 * time.Millisecond,
		OperationLabel: "test",
	}, func(ctx context.Context) (string, error) {
		return "ok", nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "ok" {
		t.Fatalf("got %q, want %q", result, "ok")
	}
}

func TestRetryDoRetryThenSuccess(t *testing.T) {
	ctx := context.Background()
	var attempts atomic.Int32

	result, err := RetryDo(ctx, RetryConfig{
		MaxRetries:     3,
		BaseDelay:      1 * time.Millisecond,
		OperationLabel: "test_retry",
	}, func(ctx context.Context) (string, error) {
		n := attempts.Add(1)
		if n < 3 {
			return "", errors.New("transient error")
		}
		return "ok", nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "ok" {
		t.Fatalf("got %q, want %q", result, "ok")
	}
	if n := attempts.Load(); n != 3 {
		t.Fatalf("expected 3 attempts, got %d", n)
	}
}

func TestRetryDoExhaustRetries(t *testing.T) {
	ctx := context.Background()
	var attempts atomic.Int32

	_, err := RetryDo(ctx, RetryConfig{
		MaxRetries:     2,
		BaseDelay:      1 * time.Millisecond,
		OperationLabel: "test_exhaust",
	}, func(ctx context.Context) (string, error) {
		attempts.Add(1)
		return "", errors.New("persistent error")
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if n := attempts.Load(); n != 3 {
		t.Fatalf("expected 3 attempts (2 retries), got %d", n)
	}
}

func TestRetryDoContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := RetryDo(ctx, RetryConfig{
		MaxRetries:     3,
		BaseDelay:      1 * time.Millisecond,
		OperationLabel: "test_cancel",
	}, func(ctx context.Context) (string, error) {
		return "", errors.New("some error")
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestRetryDoShouldRetryFilter(t *testing.T) {
	ctx := context.Background()
	permErr := errors.New("permanent error")
	var attempts atomic.Int32

	_, err := RetryDo(ctx, RetryConfig{
		MaxRetries:     3,
		BaseDelay:      1 * time.Millisecond,
		OperationLabel: "test_filter",
		ShouldRetry: func(err error) bool {
			return err.Error() != "permanent error"
		},
	}, func(ctx context.Context) (string, error) {
		attempts.Add(1)
		return "", permErr
	})
	if !errors.Is(err, permErr) {
		t.Fatalf("expected permanent error, got %v", err)
	}
	if n := attempts.Load(); n != 1 {
		t.Fatalf("expected 1 attempt, got %d", n)
	}
}

func TestRetryDoVoid(t *testing.T) {
	ctx := context.Background()
	var attempts atomic.Int32

	err := RetryDoVoid(ctx, RetryConfig{
		MaxRetries:     2,
		BaseDelay:      1 * time.Millisecond,
		OperationLabel: "test_void",
	}, func(ctx context.Context) error {
		n := attempts.Add(1)
		if n < 2 {
			return errors.New("transient error")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n := attempts.Load(); n != 2 {
		t.Fatalf("expected 2 attempts, got %d", n)
	}
}

func TestCalcBackoff(t *testing.T) {
	base := 100 * time.Millisecond
	maxDelay := 5 * time.Second

	d1 := calcBackoff(base, maxDelay, 1)
	if d1 < 50*time.Millisecond || d1 > 100*time.Millisecond {
		t.Fatalf("backoff attempt 1 out of range: %v", d1)
	}

	d2 := calcBackoff(base, maxDelay, 2)
	if d2 < 100*time.Millisecond || d2 > 200*time.Millisecond {
		t.Fatalf("backoff attempt 2 out of range: %v", d2)
	}

	d3 := calcBackoff(base, maxDelay, 6)
	if d3 > maxDelay {
		t.Fatalf("backoff attempt 6 exceeded maxDelay: %v > %v", d3, maxDelay)
	}
}

func TestDefaultShouldRetry(t *testing.T) {
	if !alwaysRetry(errors.New("anything")) {
		t.Fatal("alwaysRetry should return true for any error")
	}
}

func TestRetryDoDefaultConfigValues(t *testing.T) {
	ctx := context.Background()
	_, err := RetryDo(ctx, RetryConfig{}, func(ctx context.Context) (string, error) {
		return "", errors.New("fail")
	})
	if err == nil {
		t.Fatal("expected error")
	}
}
