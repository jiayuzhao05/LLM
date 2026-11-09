// 数组（Array）
// Go 的数组是固定长度的，长度是类型的一部分（[5]int ≠ [3]int）
package main

import "fmt"

func main() {
	// ---- 声明数组：var name [size]type ----
	// 未赋值的元素自动初始化为零值（int 的零值是 0）
	var a [5]int
	fmt.Println("emp:", a) // 输出: emp: [0 0 0 0 0]

	// 通过下标访问和赋值（下标从 0 开始）
	a[4] = 100
	fmt.Println("set:", a)    // 输出: set: [0 0 0 0 100]
	fmt.Println("get:", a[4]) // 输出: get: 100

	// len() 函数获取数组长度
	fmt.Println("len:", len(a)) // 输出: len: 5

	// ---- 数组字面量初始化 ----
	// 用花括号 {} 直接提供初始值
	b := [5]int{1, 2, 3, 4, 5}
	fmt.Println("dcl:", b) // 输出: dcl: [1 2 3 4 5]

	// ---- 二维数组 ----
	// var name [行数][列数]type
	var twoD [2][3]int // 2行3列，所有元素初始化为 0
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			twoD[i][j] = i + j // 嵌套循环赋值
		}
	}
	fmt.Println("2d: ", twoD) // 输出: 2d:  [[0 1 2] [1 2 3]]
}

// 【数组 vs Slice 的区别】
// 数组：[5]int  → 长度固定，是值类型（赋值/传参时复制整个数组）
// 切片：[]int  → 长度可变，是引用类型（赋值/传参时共享底层数组）
// 实际开发中 Slice 更常用，数组用得相对少
