// Slice 的"类指针"行为详解
// Slice 传参时传的是"Slice header"（ptr+len+cap 的结构体拷贝）
// 对已有元素的修改会影响原 Slice，但 append 后不一定
// 参考：https://medium.com/swlh/golang-tips-why-pointers-to-slices-are-useful-and-how-ignoring-them-can-lead-to-tricky-bugs-cac90f72e77b
package main

import "fmt"

// modifyVal 接收 Slice（值传递 Slice header，但底层数组是共享的）
func modifyVal(slice []string) {
	slice[0] = "Z"              // ✅ 修改已有元素：影响原 Slice（共享底层数组）
	slice = append(slice, "Y")  // ⚠️ append 可能分配新底层数组，此后 slice 指向新数组
	slice[0] = "W"              // 修改的是 append 后的新数组（如果扩容），原 Slice 不受影响
	// 注意：函数结束后，局部 slice 变量被丢弃，对 append 后的 "Y" 和 "W" 的修改都不会反映到外部
}

func main() {
	s := make([]string, 3)
	fmt.Println("emp:", s) // 输出: emp: [  ]

	s[0] = "a"
	s[1] = "b"
	s[2] = "c"
	s = append(s, "d")
	s = append(s, "e", "f")
	fmt.Println("before:", s) // 输出: before: [a b c d e f]

	modifyVal(s)
	// 分析：
	// - modifyVal 内执行 slice[0] = "Z" → s[0] 变为 "Z"（共享底层数组）
	// - append(slice, "Y") → 此时 s 的 len=6, cap 可能是 6（刚好满），
	//   append 会分配新数组，函数内的 slice 指向新数组
	// - slice[0] = "W" 修改的是新数组，不影响 s
	// - 函数返回，对新数组的所有修改都丢失
	fmt.Println("after modifyVal:", s)
	// 实际输出: after modifyVal: [Z b c d e f]
	// s[0] 变成了 "Z"（第一次修改生效），其余不变
}

// 【关键结论：Slice 传参行为】
// ✅ 修改已有元素（slice[i] = x）→ 影响原 Slice（共享底层数组）
// ❌ append 超出容量后      → 新分配底层数组，函数内的修改不影响原 Slice
// ❌ 重新赋值（slice = ...） → 只改局部变量，不影响外部
//
// 如果需要让函数内的 append 也影响外部，应传 *[]string（指向 Slice 的指针）
