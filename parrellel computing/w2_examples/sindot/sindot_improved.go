// 最优版：本地累加 + 末尾一次原子提交
// 策略：每个 goroutine 用局部变量（localsum）累加，避免频繁原子操作
//       最后用一次 atomic.AddInt32 将局部和合并到全局 sum
// 这是高性能并行求和的标准模式：本地 reduction + 最终 atomic merge
package main

import (
	"fmt"
	"sync"
	"time"
	"os"
	"strconv"
	"path/filepath"
	"math"
	"sync/atomic"
)

func dotSequential(a []int32, b []int32) int32 {
	var sum int32
	numElements := len(a)
	for i := 0; i < numElements; i++ {
		sum += int32(math.Abs(math.Sin(float64(a[i])) * float64(b[i])))
	}
	return sum
}

// dotChunk 改进版：
// 1. 用局部变量 localsum 在栈上累加（无竞态，无原子开销）
// 2. 循环结束后只做一次 atomic.AddInt32（将局部结果合并到全局 sum）
// 大幅减少了原子操作次数：从 O(numElements/numThreads) 次降低到 1 次
func dotChunk(a []int32, b []int32, sum *int32, wg *sync.WaitGroup) {
	numElements := len(a)
	var localsum int32 = 0 // 局部变量，只有本 goroutine 访问，无需同步
	for i := 0; i < numElements; i++ {
		localsum += int32(math.Abs(math.Sin(float64(a[i])) * float64(b[i]))) // 普通加法，极快
	}
	// 只在最后做一次原子操作，将本线程的局部和合并到全局 sum
	atomic.AddInt32(sum, localsum)
	wg.Done()
}

func dotParallel(a []int32, b []int32, numThreads int) int32 {
	var sum int32 = 0 // 所有线程共享的全局 sum
	numElements := len(a)
	numElementsPerThread := int(math.Ceil(float64(numElements) / float64(numThreads)))
	var wg sync.WaitGroup
	for i := 0; i < numThreads; i++ {
		chunkStart := numElementsPerThread * i
		if chunkStart > numElements {
			chunkStart = numElements
		}
		chunkEnd := chunkStart + numElementsPerThread
		if chunkEnd > numElements {
			chunkEnd = numElements
		}
		wg.Add(1)
		go dotChunk(a[chunkStart:chunkEnd], b[chunkStart:chunkEnd], &sum, &wg)
	}
	wg.Wait()
	return sum
}

func main() {
	if len(os.Args) < 3 {
		panic(fmt.Sprintf("Usage: go run %s.go <problemsize> <threadcount>\n", filepath.Base(os.Args[0])))
	}
	numElements, err := strconv.Atoi(os.Args[1])
	if err != nil {
		panic(err)
	}
	numThreads, err := strconv.Atoi(os.Args[2])
	if err != nil {
		panic(err)
	}

	a := make([]int32, numElements)
	b := make([]int32, numElements)
	for i := 0; i < numElements; i++ {
		a[i] = int32(i)
		b[i] = int32(numElements - i)
	}

	start := time.Now()
	dot_sequential := dotSequential(a[:], b[:])
	duration := time.Since(start)
	fmt.Printf("computed %d, sequential, time: %s\n", dot_sequential, duration)

	start = time.Now()
	dot_parallel := dotParallel(a[:], b[:], numThreads)
	duration = time.Since(start)
	fmt.Printf("computed %d, %d threads, time: %s\n", dot_parallel, numThreads, duration)
}

// 【五个版本性能】
// 版本               | 同步方式            | 正确性 | 性能
// ─────────────────────────────────────────────────────────
// sindot_race.go     | 无（竞态）           | ❌     | 最快（但错误）
// sindot_fs.go       | 各自 sum[i]          | ✅     | 差（False Sharing）
// sindot_nfs.go      | sum[i*16] 填充       | ✅     | 中（无 False Sharing）
// sindot_atomic.go   | 每次 atomic.Add      | ✅     | 差（原子操作竞争多）
// sindot_improved.go | 本地累加 + 末尾atomic | ✅     | 最优 ✅
//
// 【最优策略：Local Reduction Pattern（本地归约模式）】
// 1. 每个线程用局部变量累积私有结果（纯本地操作，无竞争）
// 2. 工作全部完成后，用一次原子操作将局部结果汇总到全局变量
// 原子操作次数：O(numThreads) 而非 O(numElements)
