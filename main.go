package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strconv"
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

func runBenchmark(concurrency int, args *Args) {
	rand.Seed(time.Now().UnixNano())
	ctx := context.Background()
	cli, err := client.New(ctx, &client.ClientConfig{Address: args.addr})
	if err != nil {
		panic(fmt.Errorf("failed to connect to Milvus: %v", err))
	}
	defer cli.Close(ctx)

	collectionName := "bench_" + strings.ReplaceAll(uuid.New().String(), "-", "")
	args.log("Using collection: %s (concurrency: %d)", collectionName, concurrency)

	has, err := cli.HasCollection(ctx, client.NewHasCollectionOption(collectionName))
	if err != nil {
		panic(err)
	}
	if has {
		_ = cli.DropCollection(ctx, client.NewDropCollectionOption(collectionName))
	}
	if args.clean {
		collections, _ := cli.ListCollections(ctx, client.NewListCollectionOption())
		args.log("Going to clean collections: %v", collections)
		for _, cname := range collections {
			cli.DropCollection(ctx, client.NewDropCollectionOption(cname))
		}
		collections, _ = cli.ListCollections(ctx, client.NewListCollectionOption())
		args.log("After clean: %v", collections)
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
					"dim": fmt.Sprintf("%d", args.dim),
				},
			},
		},
	}
	if err := cli.CreateCollection(ctx, client.NewCreateCollectionOption(collectionName, schema).WithShardNum(1)); err != nil {
		panic(err)
	}

	const maxLatencyBucket = 5000
	var latencyBuckets [maxLatencyBucket + 1]int64
	var (
		wg                  sync.WaitGroup
		mu                  sync.Mutex
		intervalDurations   []float64
		intervalInsertCount int64 = 0
		totalInserted       int64 = 0
		totalLatency        int64 = 0
		globalIDCounter     int64 = 0
		failedCount         int64 = 0
	)
	sem := make(chan struct{}, concurrency)
	runCtx, runCancel := context.WithCancel(ctx)
	defer runCancel()
	startAll := time.Now()

	go func() {
		time.Sleep(args.duration)
		runCancel()
	}()

	go func() {
		ticker := time.NewTicker(args.interval)
		defer ticker.Stop()
		lastTime := time.Now()
		for {
			select {
			case <-runCtx.Done():
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
					args.log("[%.0fs] No inserts this interval", time.Since(startAll).Seconds())
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
				args.log("[%.0fs] Interval Inserts: %d, RPS: %.2f, Avg: %.2fms, P99: %.2fms",
					time.Since(startAll).Seconds(), count, rps, avg, p99)
			}
		}
	}()

insertLoop:
	for {
		select {
		case <-runCtx.Done():
			break insertLoop
		default:
			sem <- struct{}{}
			wg.Add(1)
			go func() {
				defer wg.Done()
				defer func() { <-sem }()

				ids := make([]int64, args.batchSize)
				vectors := make([][]float32, args.batchSize)
				for i := 0; i < args.batchSize; i++ {
					ids[i] = atomic.AddInt64(&globalIDCounter, 1)
					vectors[i] = randomFloatVector(args.dim)
				}

				start := time.Now()
				_, err := cli.Insert(ctx, client.NewColumnBasedInsertOption(
					collectionName,
					column.NewColumnInt64("id", ids),
					column.NewColumnFloatVector("vector", args.dim, vectors),
				))
				duration := time.Since(start)
				if err == nil {
					latency := float64(duration.Milliseconds())
					atomic.AddInt64(&totalInserted, int64(args.batchSize))
					atomic.AddInt64(&intervalInsertCount, int64(args.batchSize))
					atomic.AddInt64(&totalLatency, int64(latency))
					mu.Lock()
					intervalDurations = append(intervalDurations, latency)
					bucket := int(latency)
					if bucket > maxLatencyBucket {
						bucket = maxLatencyBucket
					}
					latencyBuckets[bucket]++
					mu.Unlock()
				} else {
					atomic.AddInt64(&failedCount, 1)
					args.log("Insert failed: %v", err)
				}
			}()
		}
	}

	wg.Wait()
	total := atomic.LoadInt64(&totalInserted)
	failures := atomic.LoadInt64(&failedCount)
	elapsed := time.Since(startAll)
	avgLatency := float64(totalLatency) / float64(total/int64(args.batchSize))

	mu.Lock()
	totalCount := int64(0)
	for _, c := range latencyBuckets {
		totalCount += c
	}
	threshold := int64(float64(totalCount) * 0.99)
	cumulative := int64(0)
	p99 := -1
	for i, c := range latencyBuckets {
		cumulative += c
		if cumulative >= threshold {
			p99 = i
			break
		}
	}
	mu.Unlock()

	args.log("\n======= Final Benchmark Result for concurrency = %d =======", concurrency)
	args.log("Collection: %s", collectionName)
	args.log("Total inserted: %d", total)
	args.log("Failed inserts: %d", failures)
	args.log("Success rate: %.2f%%", 100.0*float64(total)/float64(total+failures))
	args.log("Elapsed: %.2fs", elapsed.Seconds())
	args.log("RPS: %.2f", float64(total)/elapsed.Seconds())
	args.log("Avg latency: %.2f ms", avgLatency)
	if p99 >= 0 {
		args.log("Approximate global P99 latency: %d ms", p99)
	} else {
		args.log("Global P99 latency not available")
	}
	if args.clean {
		cli.DropCollection(ctx, client.NewDropCollectionOption(collectionName))
		args.log("Drop collection %s", collectionName)
	}
}

type Args struct {
	addr      string
	dim       int
	batchSize int
	interval  time.Duration
	duration  time.Duration
	logFunc   func(format string, args ...interface{})
	logFile   *os.File
	clean     bool
}

func (a *Args) log(format string, args ...interface{}) {
	timestamp := time.Now().UTC().Format("2006-01-02T15:04:05Z07:00")
	line := fmt.Sprintf("[%s] %s", timestamp, fmt.Sprintf(format, args...))
	fmt.Println(line)
	a.logFile.WriteString(line + "\n")
}

func main() {
	addr := flag.String("addr", "localhost:19530", "Milvus server address")
	dim := flag.Int("dim", 768, "Dimension of the float vectors")
	batchSize := flag.Int("batch", 100, "Number of vectors per insert")
	concurrencyList := flag.String("concurrency", "10,50,100", "Comma-separated list of concurrent workers")
	interval := flag.Duration("interval", 10*time.Second, "Interval for metrics reporting")
	duration := flag.Duration("duration", 1*time.Minute, "Benchmark duration")
	logFile := flag.String("log", "milvus_benchmark.log", "Path to log file")
	clean := flag.Bool("clean", false, "clean all collection")
	flag.Parse()

	logF, err := os.OpenFile(*logFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		panic(fmt.Errorf("failed to open log file: %v", err))
	}
	defer logF.Close()

	args := &Args{
		addr:      *addr,
		dim:       *dim,
		batchSize: *batchSize,
		interval:  *interval,
		duration:  *duration,
		logFile:   logF,
		clean:     *clean,
	}

	args.logFunc = args.log

	concurrencyValues := strings.Split(*concurrencyList, ",")
	for _, val := range concurrencyValues {
		c, err := strconv.Atoi(strings.TrimSpace(val))
		if err != nil {
			panic(fmt.Errorf("invalid concurrency value: %v", err))
		}
		runBenchmark(c, args)
	}
}
