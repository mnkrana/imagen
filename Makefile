.PHONY: test lint build clean

test:
	go test ./... -v -count=1

test-race:
	go test ./... -race -count=1

test-integration:
	go test -tags=integration ./... -v -count=1

lint:
	golangci-lint run ./...

build:
	go build ./...

clean:
	go clean ./...

tidy:
	go mod tidy
