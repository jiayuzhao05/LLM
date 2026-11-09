// Goroutine 基础示例：并发动画 + 斐波那契计算
// 演示：go 关键字启动 goroutine，主 goroutine 和子 goroutine 并发运行
package main

import (
	"fmt"
	"time"
)

func main() {
	// go 关键字：启动一个新的 goroutine（轻量级线程）
	// spinner 在后台并发运行，主 goroutine 继续往下执行
	go spinner(2 * time.Millisecond)

	// 主 goroutine 计算第 45 个斐波那契数（耗时操作）
	// 与此同时，spinner goroutine 在终端显示旋转动画
	const n = 45
	fibN := fib(n) // 慢速递归计算（约需几秒）

	// \r 是回车符（Carriage Return），将光标移回行首，覆盖掉 spinner 的输出
	fmt.Printf("\rFibonacci(%d) = %d\n", n, fibN)
	// 主函数返回时，程序退出，spinner goroutine 也随之终止
	// 注意：Go 中主 goroutine 退出会导致所有其他 goroutine 强制终止
}

// spinner 持续打印旋转字符（-\|/），模拟加载动画
// delay time.Duration：等待时间，time.Duration 是 int64 的别名（纳秒为单位）
func spinner(delay time.Duration) {
	for {
		// range 字符串时，迭代的是 rune（Unicode 码点）
		// `-\|/` 是四个旋转字符
		for _, r := range `-\|/` {
			fmt.Printf("\r%c", r) // %c 格式化输出字符
			time.Sleep(delay)     // 暂停 delay 时间（此 goroutine 让出 CPU）
		}
	}
}

// fib 递归计算斐波那契数（故意用慢速递归来演示并发）
// 时间复杂度 O(2^n)，n=45 时约需几秒
func fib(x int) int {
	if x < 2 {
		return x
	}
	return fib(x-1) + fib(x-2)
}

// 【Goroutine 要点】
// go f(args)   → 在新 goroutine 中调用 f，主程序立即继续（非阻塞）
// goroutine 比系统线程更轻量：初始栈约 2KB，可运行数十万个
// 主 goroutine 退出 → 所有 goroutine 立即终止
// 同步方式：WaitGroup、Channel、Mutex 等（见后续文件）
//
// 【time 包常用函数】
// time.Sleep(d)       → 当前 goroutine 休眠 d 时间
// time.Millisecond    → 1ms（time.Duration 常量）
// 2 * time.Millisecond → 2ms
