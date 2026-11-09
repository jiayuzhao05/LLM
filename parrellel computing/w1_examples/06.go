// switch 语句
package main

import (
	"fmt"
	"time"
)

func main() {
	// ---- 形式1：switch（按值匹配）----
	i := 2
	fmt.Print("Write ", i, " as ")
	switch i {
	case 1:
		fmt.Println("one")
	case 2:
		fmt.Println("two") // 匹配到 case 2，输出: two
	case 3:
		fmt.Println("three")
	}
	// 注意：Go 的 switch 每个 case 自动 break（不会 fallthrough）
	// 与 C/Java 不同，不需要手写 break

	// ---- 形式2：switch 匹配复杂类型（多个值共享一个 case）----
	switch time.Now().Weekday() {
	case time.Saturday, time.Sunday: // 多个值用逗号分隔
		fmt.Println("It's the weekend")
	default: // default 分支，类似 else
		fmt.Println("It's a weekday")
	}

	// ---- 形式3：无表达式的 switch（等价于 if-else if 链）----
	t := time.Now()
	switch { // 没有表达式，每个 case 是一个独立的布尔条件
	case t.Hour() < 12:
		fmt.Println("It's before noon")
	default:
		fmt.Println("It's after noon")
	}

	// ---- 形式4：类型 switch（type switch）----
	// 用于判断接口变量的实际类型
	// interface{} 是 Go 的空接口，可以存储任意类型的值
	whatAmI := func(i interface{}) { // 匿名函数（lambda）
		switch t := i.(type) { // i.(type) 是类型断言语法，t 是实际类型的值
		case bool:
			fmt.Println("I'm a bool")
		case int:
			fmt.Println("I'm an int")
		default:
			fmt.Printf("Don't know type %T\n", t) // %T 打印变量的类型
		}
	}
	whatAmI(true)  // 输出: I'm a bool
	whatAmI(1)     // 输出: I'm an int
	whatAmI("hey") // 输出: Don't know type string
}

// 【switch 语法要点】
// 1. Go 的 switch 每个 case 自动 break，无需手写
// 2. 如需贯穿，用 fallthrough 关键字
// 3. case 可以有多个值：case 1, 2, 3:
// 4. 无表达式 switch 等价于 if-else if 链
// 5. 类型 switch：switch t := x.(type) { case int: ... }
// 6. 匿名函数：func(参数) { 函数体 }，可以直接调用
