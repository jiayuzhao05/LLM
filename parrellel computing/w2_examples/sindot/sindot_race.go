// 有数据竞争（Data Race）的并行点积计算
// 问题：多个 goroutine 同时对同一个 *sum 进行读-修改-写操作，没有同步保护
// 可用 go run -race sindot_race.go 检测竞争
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

// dotSequential 串行点积：∑ |sin(a[i]) * b[i]|
// 单线程版本，作为基准（baseline）用于对比加速比
func dotSequential(a []int32, b []int32) int32 {
	var sum int32
	numElements := len(a)
	for i := 0; i < numElements; i++ {
		// int32() 类型转换；math.Abs/Sin 需要 float64，结果再转回 int32
		sum += int32(math.Abs(math.Sin(float64(a[i])) * float64(b[i])))
	}
	return sum
}

// dotChunk 处理数组的一个分块（chunk），由 goroutine 调用
// 数据竞争：多个 goroutine 同时执行 *sum += ...（非原子操作）
// *sum += x 实际是：读取 *sum → 加 x → 写回 *sum（三步，可被中断）
// sum *int32：指向共享累加器的指针
// wg *sync.WaitGroup：用于通知主线程本 goroutine 已完成
func dotChunk(a []int32, b []int32, sum *int32, wg *sync.WaitGroup) {
	numElements := len(a)
	for i := 0; i < numElements; i++ {
		*sum += int32(math.Abs(math.Sin(float64(a[i])) * float64(b[i]))) // ❌ 竞态！
	}
	wg.Done() // 通知 WaitGroup 此 goroutine 完成（相当于 count--）
}

// dotParallel 将数组均分给 numThreads 个 goroutine 并行计算
func dotParallel(a []int32, b []int32, numThreads int) int32 {
	var sum int32 = 0
	numElements := len(a)
	// math.Ceil 向上取整，确保最后一个线程能处理剩余元素
	numElementsPerThread := int(math.Ceil(float64(numElements) / float64(numThreads)))

	var wg sync.WaitGroup // WaitGroup：等待一组 goroutine 全部完成
	for i := 0; i < numThreads; i++ {
		// 计算每个线程处理的数组分块范围 [chunkStart, chunkEnd)
		chunkStart := numElementsPerThread * i
		if chunkStart > numElements {
			chunkStart = numElements // 防止越界
		}
		chunkEnd := chunkStart + numElementsPerThread
		if chunkEnd > numElements {
			chunkEnd = numElements // 防止越界
		}

		wg.Add(1) // 注册一个待完成的 goroutine（count++）
		// a[chunkStart:chunkEnd] 是 Slice 切片，共享底层数组（零拷贝）
		// &sum 传递共享累加器的指针 ← 这是竞态的根源
		go dotChunk(a[chunkStart:chunkEnd], b[chunkStart:chunkEnd], &sum, &wg)
	}

	wg.Wait() // 阻塞，直到所有 wg.Done() 被调用（即所有 goroutine 完成）
	return sum
}

func main() {
	// os.Args 是命令行参数列表：os.Args[0] 是程序名，os.Args[1], [2]... 是用户参数
	if len(os.Args) < 3 {
		panic(fmt.Sprintf("Usage: go run %s.go <problemsize> <threadcount>\n", filepath.Base(os.Args[0])))
	}
	// strconv.Atoi 将字符串转为 int，返回 (int, error)
	numElements, err := strconv.Atoi(os.Args[1])
	if err != nil {
		panic(err) // panic 终止程序并打印错误信息
	}
	numThreads, err := strconv.Atoi(os.Args[2])
	if err != nil {
		panic(err)
	}

	// 初始化输入数组
	a := make([]int32, numElements)
	b := make([]int32, numElements)
	for i := 0; i < numElements; i++ {
		a[i] = int32(i)
		b[i] = int32(numElements - i)
	}

	// 计时：time.Now() 记录开始时间
	start := time.Now()
	dot_sequential := dotSequential(a[:], b[:]) // a[:] 等价于 a（取整个 Slice）
	duration := time.Since(start)               // time.Since 计算经过时间
	fmt.Printf("computed %d, sequential, time: %s\n", dot_sequential, duration)

	start = time.Now()
	dot_parallel := dotParallel(a[:], b[:], numThreads)
	duration = time.Since(start)
	fmt.Printf("computed %d, %d threads, time: %s\n", dot_parallel, numThreads, duration)
}

// 【sync.WaitGroup 用法】
// var wg sync.WaitGroup
// wg.Add(n)  → 增加计数 n（启动 goroutine 前调用）
// wg.Done()  → 计数减 1（goroutine 完成时调用，相当于 Add(-1)）
// wg.Wait()  → 阻塞，直到计数为 0
//
// 【数据竞态 Data Race】
// *sum += x 不是原子操作，在多线程下会产生竞态条件
// 后续文件展示三种修复方法：
//   sindot_fs.go    → 每线程单独的 sum 槽（但有 False Sharing）
//   sindot_nfs.go   → 每线程 sum 槽间填充（消除 False Sharing）
//   sindot_atomic.go → 原子操作（atomic.AddInt32）
//   sindot_improved.go → 本地累加 + 最后一次原子提交（最优）
