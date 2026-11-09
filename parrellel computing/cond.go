package main

import (
	"fmt"
	"math/rand"
	"sync"
)

type SharedContext struct {
	mutex       *sync.Mutex
	cond        *sync.Cond
	wgContext   *sync.WaitGroup
	counter     int
	threadCount int
}

func worker(goID int, ctx *SharedContext) {

	//All threads compute the fib of 20
	fib(rand.Int() % 20)

	/******* barrier *******/
	ctx.mutex.Lock() //ctx.cond.L.Lock()
	ctx.counter++
	fmt.Printf("Goroutine #%v finished.\n", goID)
	if ctx.counter == ctx.threadCount {
		ctx.cond.Broadcast()
	} else {
		for ctx.counter != ctx.threadCount {
	                fmt.Printf("Goroutine #%v goes to sleep.\n", goID)
			ctx.cond.Wait()
	                fmt.Printf("Goroutine #%v woke up.\n", goID)
		}
	}
	ctx.mutex.Unlock()

	if goID == 0 {
		fmt.Printf("Everyone finished. Yes!\n")
	}

	ctx.wgContext.Done()
}
func fib(x int) int {

	if x < 2 {
		return x
	}
	return fib(x-1) + fib(x-2)

}
func main() {

	var wg sync.WaitGroup
	threadCount := 10

	//Setup my go routine context with barrier for mutex and condition variable
	var mutex sync.Mutex
	condVar := sync.NewCond(&mutex)
	context := SharedContext{wgContext: &wg, threadCount: threadCount, cond: condVar, mutex: &mutex}

	//Spawn my threads to begin calculating fib number
	for i := 0; i < threadCount; i++ {
		wg.Add(1)
		go worker(i, &context)
	}
	wg.Wait()
	fmt.Printf("Done.\n")
}
