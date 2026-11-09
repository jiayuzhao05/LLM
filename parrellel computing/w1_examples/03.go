// 常量（const）
package main

import (
	"fmt"
	"math"
)

// const 声明包级别常量（在函数外部，整个包可见）
// 常量必须在编译期确定值，不能使用运行时计算的结果
const s string = "constant" // 显式指定类型为 string

func main() {
	fmt.Println(s) // 输出: constant

	// 函数内部也可以声明 const
	// Go 的常量可以是"无类型"（untyped），精度极高（任意精度整数/浮点）
	const n = 500000000 // 无类型整数常量

	// 常量可以参与表达式，精度在编译期保留
	const d = 3e20 / n // 3e20 = 3 × 10²⁰，结果为 6e11（无类型浮点常量）
	fmt.Println(d)     // 输出: 6e+11

	// 将无类型浮点常量转换为 int64（截断小数部分）
	fmt.Println(int64(d)) // 输出: 600000000000

	// math.Sin 需要 float64 参数，无类型常量 n 会自动转换为 float64
	fmt.Println(math.Sin(n)) // 输出: -0.28470407323754404（sin(5×10⁸)）
}

// 【常量 vs 变量】
// const：编译期确定，不可修改，可以是高精度无类型值
// var  ：运行期可变，有具体类型，可被修改
