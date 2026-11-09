// 原子操作版并行点积（每次累加都用 atomic）
// 用 sync/atomic 包的原子操作替代普通的 += 来解决竞态
// 缺点：每次迭代都需要原子操作，原子操作开销较大，性能不如 sindot_improved.go
package main

import (
	"fmt"
	"sync"
	"time"
	"os"
	"strconv"
	"path/filepath"
	"math"
	"sync/atomic" // 原子操作包
)

func dotSequential(a []int32, b []int32) int32 {
	var sum int32
	numElements := len(a)
	for i := 0; i < numElements; i++ {
		sum += int32(math.Abs(math.Sin(float64(a[i])) * float64(b[i])))
	}
	return sum
}

// dotChunk 使用 atomic.AddInt32 替代普通的 *sum += x
// atomic.AddInt32(ptr, delta) 原子性地执行 *ptr += delta，线程安全
// 原子操作保证了"读-改-写"是不可中断的硬件指令（如 x86 的 LOCK XADD）
func dotChunk(a []int32, b []int32, sum *int32, wg *sync.WaitGroup) {
	numElements := len(a)
	for i := 0; i < numElements; i++ {
		// 原子加法：线程安全，但每次循环都需要跨核同步
		atomic.AddInt32(sum, int32(math.Abs(math.Sin(float64(a[i]))*float64(b[i]))))
	}
	wg.Done()
}

// dotParallel：所有线程共享同一个 sum（用原子操作保护）
func dotParallel(a []int32, b []int32, numThreads int) int32 {
	var sum int32 = 0
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

// 【sync/atomic 常用函数】
// atomic.AddInt32(ptr *int32, delta int32) int32  → 原子加，返回新值
// atomic.AddInt64(ptr *int64, delta int64) int64
// atomic.LoadInt32(ptr *int32) int32               → 原子读
// atomic.StoreInt32(ptr *int32, val int32)         → 原子写
// atomic.CompareAndSwapInt32(ptr, old, new) bool   → CAS（比较并交换）
//
// 【性能对比（同等条件下）】
// sindot_race.go    → 最快（但结果错误，有竞态）
// sindot_atomic.go  → 每次迭代都原子操作 → 高竞争 → 较慢
// sindot_improved.go→ 本地累加 + 末尾一次原子 → 最优（下一个文件）
