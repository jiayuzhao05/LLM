// 【00.go】Go 程序的最小结构示例
// 每个 Go 文件必须属于一个 package（包）
// main 包是程序的入口包，只有 package main 才能被直接运行
package main

// import 引入标准库包
// "fmt" 提供格式化输入输出功能（类似 C 的 printf/println）
import "fmt"

// main() 是程序的入口函数，程序从这里开始执行
func main() {
	// fmt.Println 打印内容并自动换行
	fmt.Println("hello mpcs")
}
