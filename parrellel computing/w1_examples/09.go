// 指针（Pointer）
// Go 支持指针，但不支持指针运算（不能 ptr++ 等）
// 指针用于在函数间共享和修改变量
package main

import "fmt"

// zeroval 接收 int 的值（值传递）
// 函数内修改 ival 不影响调用者的变量（拷贝一份）
func zeroval(ival int) {
	ival = 0 // 只修改了局部拷贝，调用者的值不变
}

// zeroptr 接收 *int（int 的指针，传递引用）
// *int 表示"指向 int 的指针"类型
// 通过指针修改可以影响调用者的原始变量
func zeroptr(iptr *int) {
	*iptr = 0 // * 是解引用操作符，修改指针指向的内存中的值
}

func main() {
	i := 1
	fmt.Println("initial:", i) // 输出: initial: 1

	// 值传递：zeroval 不能修改 i
	zeroval(i)
	fmt.Println("zeroval:", i) // 输出: zeroval: 1（i 未变）

	// 指针传递：&i 取 i 的地址（&是取地址操作符）
	// zeroptr 通过指针修改了 i 的值
	zeroptr(&i)
	fmt.Println("zeroptr:", i) // 输出: zeroptr: 0（i 被修改为 0）

	// &i 打印 i 的内存地址（十六进制）
	fmt.Println("pointer:", &i) // 输出: pointer: 0xc0000b4008（每次运行不同）
}

// 【指针语法】
// *T     → 类型：指向 T 的指针，例如 *int, *string
// &x     → 取地址：获取变量 x 的内存地址，返回 *T
// *ptr   → 解引用：获取/修改指针指向的值
//
// 【值传递 vs 指针传递】
// 值传递：函数接收变量的拷贝，修改不影响原变量
// 指针传递：函数接收变量的地址，通过解引用修改原变量（适合需要修改或大型结构体）
// 注意：Go 的 Slice、Map 本身已经是引用类型，传递时无需额外取地址
