// 条件判断（if / else if / else）
package main

import "fmt"

func main() {
	// ---- 基本 if-else ----
	if 7%2 == 0 {
		fmt.Println("7 is even")
	} else {
		fmt.Println("7 is odd") // 输出: 7 is odd
	}
	// 注意：Go 的 if 条件不需要括号（()），但花括号 {} 是必须

	// ---- 只有 if（没有 else）----
	if 8%4 == 0 {
		fmt.Println("8 is divisible by 4") 
	}

	// ---- if 的初始化语句（Go 特有语法）----
	// if 初始化语句; 条件 { ... }
	// num := 9 只在这个 if-else 块的作用域内有效
	if num := 9; num < 0 {
		fmt.Println(num, "is negative")
	} else if num < 10 {
		fmt.Println(num, "has 1 digit") // 输出: 9 has 1 digit
	} else {
		fmt.Println(num, "has multiple digits")
	}
	// 这里 num 已经超出作用域，无法访问
}

// 【if 语法】
// 1. 条件不加括号：if x > 0 {}  而非 if (x > 0) {}
// 2. 花括号 {} 必须存在，且 { 必须和 if 在同一行
// 3. if 可以有初始化语句：if x := f(); x > 0 {}（x 只在该 if 块内有效）
