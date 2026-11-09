// 变量声明的多种方式
// 注意：此文件包含若干故意展示的错误示例，无法直接编译通过
package main

import "fmt"

func main() {
	// 方式1：var 关键字 + 类型推断（Go 自动推断类型为 string）
	var a = "initial"
	fmt.Println(a) // 输出: initial

	// 方式2：var 同时声明多个同类型变量，并赋初值
	var b, c int = 1, 2
	fmt.Println(b, c) // 输出: 1 2

	// 方式3：var + 类型推断（推断为 bool）
	var d = true
	fmt.Println(d) // 输出: true

	// 方式4：先声明后赋值（错误示例：Go 不允许 var e 不带类型和初始值）
	// 正确写法应为：var e int  或  var e = 5
	var e  // ← 编译错误：missing type in var declaration
	e = 5
	fmt.Println(e)

	// 错误示例：e 被重复声明
	var e int  // ← 编译错误：no new variables on left side of :=
	// 错误示例：f 是 string 但被赋值为 int
	f := "apple" // := 是短变量声明（声明 + 赋值合一，自动推断类型为 string）
	f = 2        // ← 编译错误：cannot use 2 (untyped int) as type string
	fmt.Println(f)
}

// 【变量声明方式总结】
// var name type          → 声明变量，使用零值（int=0, string="", bool=false）
// var name = value       → 声明 + 赋值，类型自动推断
// var name type = value  → 声明 + 赋值，显式指定类型
// name := value          → 短变量声明（只能在函数内部使用），最常用
// var a, b int = 1, 2    → 多变量同时声明
