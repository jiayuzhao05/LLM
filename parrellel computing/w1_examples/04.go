// 循环（for）
// Go 只有一种循环关键字：for（没有 while、do-while）
package main

import "fmt"

func main() {
	// ---- 形式1：类似 while 循环 ----
	// for 条件 { ... }
	i := 1
	for i <= 3 {
		fmt.Println(i)
		i = i + 1 // 手动递增（也可以写 i++）
	}
	// 输出: 1 2 3

	// ---- 形式2：经典三段式 for（类似 C 的 for 循环）----
	// for 初始化; 条件; 后置操作 { ... }
	for j := 7; j <= 9; j++ { // j++ 等价于 j = j + 1
		fmt.Println(j)
	}
	// 输出: 7 8 9

	// ---- 形式3：无限循环（类似 while(true)）----
	// for { ... }  没有条件 = 永远循环
	for {
		fmt.Println("loop")
		break // break 立即退出循环
	}
	// 输出: loop（只打印一次，因为立即 break）

	// ---- 形式4：带 continue 的循环 ----
	// continue 跳过当前迭代，进入下一次循环
	for n := 0; n <= 5; n++ {
		if n%2 == 0 { // 如果 n 是偶数，跳过
			continue
		}
		fmt.Println(n) // 只打印奇数
	}
	// 输出: 1 3 5
}

// 【for 循环】
// for 条件 {}              → while 循环
// for init; cond; post {}  → 经典 C 风格 for
// for {}                   → 无限循环（infinite loop）
// break    → 退出循环
// continue → 跳过本次迭代
