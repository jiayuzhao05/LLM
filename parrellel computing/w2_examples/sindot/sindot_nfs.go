// 无 False Sharing 的并行点积（No False Sharing）
// 通过在每个 sum 槽之间填充（padding）16 个 int32 的间距，确保不同线程的 sum 槽位于不同的缓存行，消除伪共享
package main

import (
	"fmt"
	"sync"
	"time"
	"os"
	"strconv"
	"path/filepath"
	"math"
)

func dotSequential(a []int32, b []int32) int32 {
	var sum int32
	numElements := len(a)
	for i := 0; i < numElements; i++ {
		sum += int32(math.Abs(math.Sin(float64(a[i])) * float64(b[i])))
	}
	return sum
}

func dotChunk(a []int32, b []int32, sum *int32, wg *sync.WaitGroup) {
	numElements := len(a)
	for i := 0; i < numElements; i++ {
		*sum += int32(math.Abs(math.Sin(float64(a[i])) * float64(b[i])))
	}
	wg.Done()
}

func dotParallel(a []int32, b []int32, numThreads int) int32 {
	// 关键：分配 numThreads * 16 个 int32 的数组
	// 线程 i 使用 sum[i*16]，相邻线程的槽间隔 16 * 4 = 64 字节 = 1 个完整缓存行
	// 每个线程的 sum 槽在独立的缓存行，修改不会影响其他线程的缓存
	sum := make([]int32, numThreads*16)

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
		// &sum[i*16]：步长为 16，确保每个槽位于不同缓存行
		go dotChunk(a[chunkStart:chunkEnd], b[chunkStart:chunkEnd], &sum[i*16], &wg)
	}
	wg.Wait()

	// 汇总时也使用 i*16 步长
	for i := 1; i < numThreads; i++ {
		sum[0] += sum[i*16]
	}
	return sum[0]
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

// 【缓存行填充技术（Cache Line Padding）】
// 缓存行大小 = 64 字节，int32 = 4 字节，所以 64/4 = 16 个 int32 占一个缓存行
//
// sindot_fs.go:  sum[i]    → 步长 1（4字节），多个线程的数据在同一缓存行 → False Sharing
// sindot_nfs.go: sum[i*16] → 步长 16（64字节），每个线程独占一个缓存行 → 无 False Sharing
//
// 内存布局示意（4个线程）：
// fs:  [sum0|sum1|sum2|sum3|...(同一缓存行)]
// nfs: [sum0|pad|pad|...|pad||sum1|pad|pad|...|pad||sum2|...] 每组64字节
//
// 代价：内存使用增加 16 倍，但性能通常显著提升
