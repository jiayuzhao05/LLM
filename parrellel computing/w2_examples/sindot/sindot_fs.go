// 有 False Sharing（伪共享）的并行点积
// 修复了数据竞态：每个线程写自己专属的 sum[i]，互不覆盖
// 但引入了新问题：False Sharing（伪共享），性能较差
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

// dotChunk 现在写的是 &sum[i]（线程 i 专属的 sum 槽），无竞态
// 但 sum[0], sum[1], sum[2]... 在内存中相邻，可能在同一缓存行（cache line）
func dotChunk(a []int32, b []int32, sum *int32, wg *sync.WaitGroup) {
	numElements := len(a)
	for i := 0; i < numElements; i++ {
		*sum += int32(math.Abs(math.Sin(float64(a[i])) * float64(b[i])))
	}
	wg.Done()
}

// dotParallel：给每个线程分配 sum 数组中的独立槽位
func dotParallel(a []int32, b []int32, numThreads int) int32 {
	// False Sharing：sum[0..numThreads-1] 连续存储
	// int32 是 4 字节，缓存行通常是 64 字节（可存 16 个 int32）
	// 如果 numThreads <= 16，所有线程的 sum 槽在同一缓存行！
	// 线程 A 修改 sum[0] → 导致线程 B 的缓存中 sum[1] 所在行失效
	// → 线程 B 必须重新从内存加载，即使它只访问 sum[1]
	// "伪共享"：逻辑上不共享数据，但物理上共享缓存行
	sum := make([]int32, numThreads)

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
		// &sum[i] 取 sum 数组第 i 个元素的地址，传给 goroutine
		go dotChunk(a[chunkStart:chunkEnd], b[chunkStart:chunkEnd], &sum[i], &wg)
	}
	wg.Wait()

	// 串行汇总各线程的局部和
	for i := 1; i < numThreads; i++ {
		sum[0] += sum[i]
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

// 【False Sharing（伪共享）原理】
// CPU 缓存以"缓存行"（Cache Line，通常 64 字节）为单位加载和失效
// 即使两个线程访问不同的变量，如果它们在同一缓存行，
// 一个线程的写操作会导致另一个线程的缓存行失效（MESI 协议）
// → 被迫从内存重新加载，造成大量额外的内存流量和延迟
//
// 本文件：sum[i] 相邻（间隔 4 字节），极易发生 False Sharing
// 解决方案见 sindot_nfs.go：使用 sum[i*16] 增大间隔，确保每个槽在不同缓存行
