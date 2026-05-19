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
	fmt.Printf("  Path: %s\n", result.ObjectPath)
}
