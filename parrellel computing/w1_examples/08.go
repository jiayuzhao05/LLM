// 切片（Slice）
// Slice 是 Go 中最常用的数据结构，是对底层数组的动态视图
// Slice 内部由三个字段组成：指针（ptr）、长度（len）、容量（cap）
package main

import "fmt"

func main() {
	// ---- 创建 Slice：make([]type, length) ----
	// make 分配内存并初始化，所有元素为零值（string 零值为 ""）
	s := make([]string, 3)
	fmt.Println("emp:", s) // 输出: emp: [  ]（3个空字符串）

	// 通过下标赋值
	s[0] = "a"
	s[1] = "b"
	s[2] = "c"
	fmt.Println("set:", s)    // 输出: set: [a b c]
	fmt.Println("get:", s[2]) // 输出: get: c

	fmt.Println("len:", len(s)) // 输出: len: 3

	// ---- append：向 Slice 末尾追加元素 ----
	// append 返回新的 Slice（可能指向新分配的底层数组）
	// 必须将结果赋回原变量！
	s = append(s, "d")
	s = append(s, "e", "f") // 可以一次 append 多个元素
	fmt.Println("apd:", s)  // 输出: apd: [a b c d e f]

	// ---- copy：深拷贝 Slice ----
	// copy(dst, src)，拷贝 min(len(dst), len(src)) 个元素
	c := make([]string, len(s)) // 创建与 s 等长的新 Slice
	copy(c, s)
	fmt.Println("cpy:", c) // 输出: cpy: [a b c d e f]（独立副本）

	// ---- 切片操作（slicing）：s[low:high] ----
	// 返回 s 中下标 [low, high) 的子切片（左闭右开）
	// 子切片与原 Slice 共享底层数组！
	l := s[2:5]           // 下标 2,3,4（不含5）
	fmt.Println("sl1:", l) // 输出: sl1: [c d e]

	l = s[:5]             // 从头到下标4（省略 low = 从0开始）
	fmt.Println("sl2:", l) // 输出: sl2: [a b c d e]

	l = s[2:]             // 从下标2到末尾（省略 high = 到结尾）
	fmt.Println("sl3:", l) // 输出: sl3: [c d e f]

	// ---- Slice 字面量初始化 ----
	t := []string{"g", "h", "i"} // 直接用 {} 初始化（不写长度）
	fmt.Println("dcl:", t)        // 输出: dcl: [g h i]

	// ---- 二维 Slice（不规则/锯齿形）----
	twoD := make([][]int, 3) // 外层：3个元素，每个元素是 []int
	for i := 0; i < 3; i++ {
		innerLen := i + 1               // 每行长度不同（1, 2, 3）
		twoD[i] = make([]int, innerLen) // 内层：动态长度
		for j := 0; j < innerLen; j++ {
			twoD[i][j] = i + j
		}
	}
	fmt.Println("2d: ", twoD) // 输出: 2d:  [[0] [1 2] [2 3 4]]
}

// 【Slice 要点】
// make([]T, n)     → 创建长度为 n 的 Slice，元素为零值
// append(s, v...)  → 追加元素，必须赋回：s = append(s, x)
// copy(dst, src)   → 深拷贝，互不影响
// s[low:high]      → 子切片，共享底层数组（修改会互相影响！）
// len(s)           → 长度；cap(s) → 容量
