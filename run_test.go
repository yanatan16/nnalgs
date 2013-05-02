package nnalgs

import (
	"testing"
	"math"
	"fmt"
	"math/rand"
)

var nt, nv int = 10000, 1000

func Xor(a, b float64) float64 {
	if a == b {
		return 0
	}
	return 1
}

func SimpleQuadratic(a, b float64) float64 {
	return 2 * a * a + b * b + 3 * a * b - a - b + 4
}

func DataStream(fn func(a,b float64) float64) chan Pair {
	ch := make(chan Pair)
	go func () {
		for {
			a, b := math.Floor(rand.Float64()+.5), math.Floor(rand.Float64()+.5)
			c := fn(a, b)
			p := Pair{X: []float64{a, b}, Z: c}
			ch <- p
		}
	}()
	return ch
}

func TestXor(t *testing.T) {
	network := newNetwork(2, []int{2, 1}, HyperbolicSigmoid(true), 1)
	ak := StandardAk(1, float64(nt/10), .501)
	data := DataStream(Xor)
	Backpropogation(network, data, ak, nt)

	mu, mse, sigma := PredictionErrors(network, data, nv)

	fmt.Printf("Backprop training: mu %f, mse %f, sigma %f\n", mu, mse, sigma)

	if mu > .001 {
		t.Error("Xor training failed!", mu, mse, sigma)
	}
}

func TestQuadNewton(t *testing.T) {
	network := newNetwork(2, []int{3, 1}, HyperbolicSigmoid(true), 1)
	ak := StandardAk(1, float64(nt/10), .501)
	data := DataStream(SimpleQuadratic)
	Backpropogation(network, data, ak, nt)
	fmt.Println(PredictionErrors(network, data, nv))
	// NewtonRaphsonBackpropogation(network, data, 100)
	mu, mse, sigma := PredictionErrors(network, data, nv)

	fmt.Printf("Newton Backprop training: mu %f, mse %f, sigma %f\n", mu, mse, sigma)

	if math.Abs(mu) > .001 || math.IsNaN(mu) {
		t.Error("Xor training failed!", mu, mse, sigma)
	}

}

func TestHess(t *testing.T) {
	// network := newNetwork(2, []int{2, 1}, HyperbolicSigmoid(true), 1)
	// ak := StandardAk(1, float64(nt/10), .501)
	// data := DataStream(SimpleQuadratic)
	// Backpropogation(network, data, ak, nt)

}