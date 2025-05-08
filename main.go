package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	client "github.com/milvus-io/milvus/client/v2/milvusclient"
)

func randomFloatVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()
	}
	return vec
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Command-line flags
	addr := flag.String("addr", "localhost:19530", "Milvus server address")
	dim := flag.Int("dim", 768, "Dimension of the float vectors")
	batchSize := flag.Int("batch", 100, "Number of vectors per insert")
	concurrency := flag.Int("concurrency", 10, "Number of concurrent workers")
	interval := flag.Duration("interval", 10*time.Second, "Interval for metrics reporting")
	duration := flag.Duration("duration", 1*time.Hour, "Benchmark duration")
	logFile := flag.String("log", "milvus_benchmark.log", "Path to log file")
	cleanAll := flag.Bool("clean", false, "If drop all existed collection before test")
	flag.Parse()

	logF, err := os.OpenFile(*logFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		panic(fmt.Errorf("failed to open log file: %v", err))
	}
	defer logF.Close()
	log := func(format string, args ...interface{}) {
		timestamp := time.Now().UTC().Format("2006-01-02 15:04:05.000Z")
		line := fmt.Sprintf("[%s] %s", timestamp, fmt.Sprintf(format, args...))
		fmt.Println(line)
		logF.WriteString(line + "\n")
	}

	ctx := context.Background()

	cli, err := client.New(ctx, &client.ClientConfig{Address: *addr})
	if err != nil {
		panic(fmt.Errorf("failed to connect to Milvus: %v", err))
	}
	defer cli.Close(ctx)

	// Create random collection name
	collectionName := "bench_" + strings.ReplaceAll(uuid.New().String(), "-", "")
	log("Using collection: %s", collectionName)

	// Drop if exists (precautionary)
	has, err := cli.HasCollection(ctx, client.NewHasCollectionOption(collectionName))
	if err != nil {
		panic(err)
	}
	if has {
		_ = cli.DropCollection(ctx, client.NewDropCollectionOption(collectionName))
	}
	if *cleanAll {
		collections, err := cli.ListCollections(ctx, client.NewListCollectionOption())
		if err != nil {
			panic(err)
		}
		for _, coll := range collections {
			err = cli.DropCollection(ctx, client.NewDropCollectionOption(coll))
			if err != nil {
				panic(err)
			}
			log("Dropped collection: %s", coll)
		}
	}

	schema := &entity.Schema{
		CollectionName: collectionName,
		Description:    "Benchmark collection",
		Fields: []*entity.Field{
			{
				Name:       "id",
				DataType:   entity.FieldTypeInt64,
				PrimaryKey: true,
				AutoID:     false,
			},
			{
				Name:     "vector",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": fmt.Sprintf("%d", *dim),
				},
			},
		},
	}
	if err := cli.CreateCollection(ctx, client.NewCreateCollectionOption(collectionName, schema).WithShardNum(1)); err != nil {
		panic(err)
	}

	var (
		wg                  sync.WaitGroup
		mu                  sync.Mutex
		intervalDurations   []float64
		intervalInsertCount int64 = 0
		totalInserted       int64 = 0
		totalLatency        int64 = 0
		globalIDCounter     int64 = 0
	)
	sem := make(chan struct{}, *concurrency)
	ctx, cancel := context.WithTimeout(ctx, *duration)
	defer cancel()
	startAll := time.Now()

	// Metrics goroutine
	go func() {
		ticker := time.NewTicker(*interval)
		defer ticker.Stop()
		lastTime := time.Now()
		for {
			select {
			case <-ctx.Done():
				return
			case now := <-ticker.C:
				deltaTime := now.Sub(lastTime).Seconds()
				lastTime = now

				mu.Lock()
				snapshot := append([]float64(nil), intervalDurations...)
				intervalDurations = intervalDurations[:0]
				count := atomic.SwapInt64(&intervalInsertCount, 0)
				mu.Unlock()

				if len(snapshot) == 0 {
					log("[%.0fs] No inserts this interval", time.Since(startAll).Seconds())
					continue
				}
				sort.Float64s(snapshot)
				sum := 0.0
				for _, v := range snapshot {
					sum += v
				}
				avg := sum / float64(len(snapshot))
				p99 := snapshot[int(0.99*float64(len(snapshot)))-1]
				rps := float64(count) / deltaTime
				log("[%.0fs] Interval Inserts: %d, RPS: %.2f, Avg: %.2fms, P99: %.2fms",
					time.Since(startAll).Seconds(),
					count,
					rps,
					avg, p99)
			}
		}
	}()

	// Insert loop
	for {
		select {
		case <-ctx.Done():
			goto finish
		default:
			sem <- struct{}{}
			wg.Add(1)
			go func() {
				defer wg.Done()
				defer func() { <-sem }()

				ids := make([]int64, *batchSize)
				vectors := make([][]float32, *batchSize)
				for i := 0; i < *batchSize; i++ {
					ids[i] = atomic.AddInt64(&globalIDCounter, 1)
					vectors[i] = randomFloatVector(*dim)
				}

				start := time.Now()
				_, err := cli.Insert(ctx, client.NewColumnBasedInsertOption(
					collectionName,
					column.NewColumnInt64("id", ids),
					column.NewColumnFloatVector("vector", *dim, vectors),
				))
				duration := time.Since(start)
				if err == nil {
					latency := float64(duration.Milliseconds())
					atomic.AddInt64(&totalInserted, int64(*batchSize))
					atomic.AddInt64(&intervalInsertCount, int64(*batchSize))
					atomic.AddInt64(&totalLatency, int64(latency))
					mu.Lock()
					intervalDurations = append(intervalDurations, latency)
					mu.Unlock()
				} else {
					log("Insert failed: %v", err)
				}
			}()
		}
	}

finish:
	wg.Wait()
	total := atomic.LoadInt64(&totalInserted)
	elapsed := time.Since(startAll)
	avgLatency := float64(totalLatency) / float64(total/int64(*batchSize))
	log("\n======= Final Benchmark Result =======")
	log("Collection: %s", collectionName)
	log("Total inserted: %d", total)
	log("Elapsed: %.2fs", elapsed.Seconds())
	log("RPS: %.2f", float64(total)/elapsed.Seconds())
	log("Avg latency: %.2f ms", avgLatency)
	log("Note: P99 shown only in interval logs.")
}
