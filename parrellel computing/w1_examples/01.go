// 基本运算：字符串、算术、布尔
package main

import "fmt"

func main() {
	// 字符串拼接：Go 用 + 连接字符串（与 Java/Python 相同）
	fmt.Println("go" + "lang") // 输出: golang

	// 算术运算：整数与浮点数
	fmt.Println("1+1 =", 1+1)       // 整数加法，输出: 1+1 = 2
	fmt.Println("7.0/3.0 =", 7.0/3.0) // 浮点除法，输出: 7.0/3.0 = 2.3333...
	// fmt.Println("7/3 =", 7/3)    // 整数除法会截断小数，结果为 2（已注释掉）

	// 布尔运算
	fmt.Println(true && false) // 逻辑与 AND，输出: false
	fmt.Println(true || false) // 逻辑或 OR，输出: true
	fmt.Println(!true)         // 逻辑非 NOT，输出: false
}
