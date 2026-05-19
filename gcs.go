package imagen

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/url"
	"time"

	"cloud.google.com/go/storage"
	"github.com/google/uuid"
)

// GCSStorage uploads images to Google Cloud Storage.
type GCSStorage struct {
	bucket     string
	objectPath func(*Result) string
	publicURL  func(string, string, string) string
}

// GCSOption configures the GCS storage backend.
type GCSOption func(*GCSStorage)

// WithGCSObjectNamer sets a custom function for generating GCS object paths.
func WithGCSObjectNamer(fn func(*Result) string) GCSOption {
	return func(s *GCSStorage) { s.objectPath = fn }
}

// WithGCSPublicURL sets a custom function for generating the public URL.
func WithGCSPublicURL(fn func(bucket, objectPath, token string) string) GCSOption {
	return func(s *GCSStorage) { s.publicURL = fn }
}

// NewGCSStorage creates a new GCS storage backend for the given bucket.
func NewGCSStorage(bucket string, opts ...GCSOption) *GCSStorage {
	s := &GCSStorage{
		bucket: bucket,
		objectPath: func(r *Result) string {
			ext := ".png"
			switch r.ContentType {
			case "image/png":
				ext = ".png"
			case "image/jpeg":
				ext = ".jpg"
			case "image/webp":
				ext = ".webp"
			}
			return fmt.Sprintf("images/img_%d%s", time.Now().UnixNano(), ext)
		},
		publicURL: func(bucket, objectPath, token string) string {
			encodedPath := url.PathEscape(objectPath)
			return fmt.Sprintf(
				"https://firebasestorage.googleapis.com/v0/b/%s/o/%s?alt=media&token=%s",
				bucket, encodedPath, token,
			)
		},
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Upload uploads an image result to GCS and returns the storage result.
func (s *GCSStorage) Upload(ctx context.Context, result *Result) (*StorageResult, error) {
	if s.bucket == "" {
		return nil, fmt.Errorf("gcs: %w", ErrGCSBucketRequired)
	}

	client, err := storage.NewClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("gcs: create client: %w", err)
	}
	defer client.Close()

	path := s.objectPath(result)
	token := uuid.NewString()

	w := client.Bucket(s.bucket).Object(path).NewWriter(ctx)
	w.ContentType = result.ContentType
	w.CacheControl = "private, max-age=31536000, immutable"
	w.Metadata = map[string]string{"firebaseStorageDownloadTokens": token}

	if _, err := w.Write(result.Data); err != nil {
		return nil, fmt.Errorf("gcs: write: %w", err)
	}
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("gcs: close: %w", err)
	}

	downloadURL := s.publicURL(s.bucket, path, token)

	log.Printf("Uploaded to GCS: %s (%d bytes)", downloadURL, len(result.Data))

	return &StorageResult{
		URL:         downloadURL,
		Bucket:      s.bucket,
		ObjectPath:  path,
		ContentType: result.ContentType,
		Size:        int64(len(result.Data)),
		CreatedAt:   time.Now(),
	}, nil
}

// UploadReader uploads image data from a reader to GCS and returns the storage result.
func (s *GCSStorage) UploadReader(ctx context.Context, objectPath, contentType string, r io.Reader) (*StorageResult, error) {
	if s.bucket == "" {
		return nil, fmt.Errorf("gcs: %w", ErrGCSBucketRequired)
	}

	client, err := storage.NewClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("gcs: create client: %w", err)
	}
	defer client.Close()

	token := uuid.NewString()

	w := client.Bucket(s.bucket).Object(objectPath).NewWriter(ctx)
	w.ContentType = contentType
	w.CacheControl = "private, max-age=31536000, immutable"
	w.Metadata = map[string]string{"firebaseStorageDownloadTokens": token}

	if _, err := io.Copy(w, r); err != nil {
		return nil, fmt.Errorf("gcs: upload: %w", err)
	}
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("gcs: close: %w", err)
	}

	downloadURL := s.publicURL(s.bucket, objectPath, token)

	return &StorageResult{
		URL:         downloadURL,
		Bucket:      s.bucket,
		ObjectPath:  objectPath,
		ContentType: contentType,
		CreatedAt:   time.Now(),
	}, nil
}

// EnsureFirebaseToken ensures a GCS object has a Firebase download token
// and returns the Firebase Storage download URL.
func EnsureFirebaseToken(ctx context.Context, bucket, objectPath string) (string, error) {
	client, err := storage.NewClient(ctx)
	if err != nil {
		return "", fmt.Errorf("gcs: client: %w", err)
	}
	defer client.Close()

	obj := client.Bucket(bucket).Object(objectPath)
	attrs, err := obj.Attrs(ctx)
	if err != nil {
		return "", fmt.Errorf("gcs: get attrs: %w", err)
	}

	tokens, ok := attrs.Metadata["firebaseStorageDownloadTokens"]
	if ok && tokens != "" {
		first := tokens
		for i := 0; i < len(tokens); i++ {
			if tokens[i] == ',' {
				first = tokens[:i]
				break
			}
		}
		return firebaseDownloadURL(bucket, objectPath, first), nil
	}

	newToken := uuid.NewString()
	_, err = obj.Update(ctx, storage.ObjectAttrsToUpdate{
		Metadata: map[string]string{"firebaseStorageDownloadTokens": newToken},
	})
	if err != nil {
		return "", fmt.Errorf("gcs: update metadata: %w", err)
	}

	return firebaseDownloadURL(bucket, objectPath, newToken), nil
}

func firebaseDownloadURL(bucket, objectPath, token string) string {
	encodedPath := url.PathEscape(objectPath)
	return fmt.Sprintf(
		"https://firebasestorage.googleapis.com/v0/b/%s/o/%s?alt=media&token=%s",
		bucket, encodedPath, token,
	)
}
