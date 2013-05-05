// neuralnet provides an simple Neural Network library with access to gradients and Hessians.
package nnalgs

import (
	"fmt"
	matrix "github.com/skelterjohn/go.matrix"
	"math"
	"math/rand"
)

var _ = fmt.Printf

// A scalar function for use in a neural network
type Function interface {
	// Evaluate the function
	F(float64) float64
	// Evaluate the first derivative or gradient
	G(float64) float64
	// Evaluate the second derivative or Hessian
	H(float64) float64
}

// A node in the neural network
type Node interface {
	// Evaluate the node
	F(inputs *matrix.DenseMatrix) float64
	// Evaluate the Gradient. Takes input of the inputs and gradients of the layer before it.
	G(inputs *matrix.DenseMatrix, gradients []*matrix.DenseMatrix) *matrix.DenseMatrix
	// Evaluate the Hessian. Takes input of the inputs, gradients, and hessians of the layer before it.
	H(inputs *matrix.DenseMatrix, gradients, hessians []*matrix.DenseMatrix) *matrix.DenseMatrix
	// Get the weight vector of this node
	W() *matrix.DenseMatrix
	// Set the weight vector for this node
	SetW(*matrix.DenseMatrix)
}

// A layer in the neural network
type Layer interface {
	// Evaluate the layer with network inputs as parameters and outputs as returns
	F(inputs *matrix.DenseMatrix) *matrix.DenseMatrix
	// Evaluate the gradients of each node
	G(inputs *matrix.DenseMatrix, gradients []*matrix.DenseMatrix) []*matrix.DenseMatrix
	// Evaluate the Hessian of each node
	H(inputs *matrix.DenseMatrix, gradients, hessians []*matrix.DenseMatrix) []*matrix.DenseMatrix
	// Get the weight vectors of this layer
	W() *matrix.DenseMatrix
	// Set the weight vector for this layer
	SetW(*matrix.DenseMatrix)
}

type network struct {
	layers  []layer
	ninputs int
}

func NewNetwork(ninputs int, nnodes []int, f Function, bound float64) *network {
	if nnodes[len(nnodes)-1] != 1 {
		panic("Only built for one output!")
	}
	ntwk := &network{ninputs: ninputs}
	for _, nodes := range nnodes[:len(nnodes)-1] {
		ntwk.layers = append(ntwk.layers, newLayer(nodes, ninputs+1, f, bound))
		ninputs = nodes
	}
	ntwk.layers = append(ntwk.layers, newLayer(nnodes[len(nnodes)-1], ninputs+1, Identity(true), bound))
	return ntwk
}
func (n *network) F(inputs []float64) float64 {
	x := matrix.MakeDenseMatrix(append(inputs, 1), len(inputs)+1, 1)
	for _, lyr := range n.layers {
		x = lyr.F(x)
		x = ensure(x.Transpose().Augment(matrix.Ones(1, 1))).Transpose()
	}
	return x.Get(0, 0)
}
func (n *network) FG(inputs []float64) (float64, *matrix.DenseMatrix) {
	x := matrix.MakeDenseMatrix(append(inputs, 1), len(inputs)+1, 1)
	g := make([]*matrix.DenseMatrix, 0)
	for _, lyr := range n.layers {
		x, g = lyr.F(x), lyr.G(x, g)
		x = ensure(x.Transpose().Augment(matrix.Ones(1, 1))).Transpose()
	}
	return x.Get(0, 0), g[0]
}
func (n *network) FGH(inputs []float64) (float64, *matrix.DenseMatrix, *matrix.DenseMatrix) {
	x := matrix.MakeDenseMatrix(append(inputs, 1), len(inputs)+1, 1)
	g := make([]*matrix.DenseMatrix, 0)
	h := make([]*matrix.DenseMatrix, 0)
	for _, lyr := range n.layers {
		x, g, h = lyr.F(x), lyr.G(x, g), lyr.H(x, g, h)
		x = ensure(x.Transpose().Augment(matrix.Ones(1, 1))).Transpose()
	}
	return x.Get(0, 0), g[0], h[0]
}
func (n *network) W() *matrix.DenseMatrix {
	weights := matrix.Zeros(1, 0)
	for _, lyr := range n.layers {
		weights = ensure(weights.Augment(lyr.W().Transpose()))
	}
	return weights.Transpose()
}
func (n *network) SetW(weights *matrix.DenseMatrix) {
	nin := n.ninputs
	off := 0
	for _, lyr := range n.layers {
		lw := (nin + 1) * len(lyr)
		lyr.SetW(weights.GetMatrix(off, 0, lw, 1))
		off += lw
		nin = len(lyr)
	}
}
func (n *network) Len() int {
	nin, tot := n.ninputs, 0
	for _, lyr := range n.layers {
		tot += (nin + 1) * len(lyr)
		nin = len(lyr)
	}
	return tot
}

type layer []Node

func newLayer(nodes, inputs int, f Function, bound float64) layer {
	l := make(layer, nodes)
	for i := 0; i < nodes; i++ {
		l[i] = newNode(f, initWeights(inputs, bound))
	}
	return l
}
func (lyr layer) F(inputs *matrix.DenseMatrix) *matrix.DenseMatrix {
	output := matrix.Zeros(len(lyr), 1)
	for i, node := range lyr {
		output.Set(i, 0, node.F(inputs))
	}
	return output
}
func (lyr layer) G(inputs *matrix.DenseMatrix, gradients []*matrix.DenseMatrix) []*matrix.DenseMatrix {
	output := make([]*matrix.DenseMatrix, len(lyr))
	for i, node := range lyr {
		output[i] = node.G(inputs, gradients)
	}
	return output
}
func (lyr layer) H(inputs *matrix.DenseMatrix, gradients, hessians []*matrix.DenseMatrix) []*matrix.DenseMatrix {
	output := make([]*matrix.DenseMatrix, len(lyr))
	for i, node := range lyr {
		output[i] = node.H(inputs, gradients, hessians)
	}
	return output

}
func (lyr layer) W() *matrix.DenseMatrix {
	ret := matrix.Zeros(1, 0)
	for _, node := range lyr {
		ret = ensure(ret.Augment(node.W().Transpose()))
	}
	return ret.Transpose()
}
func (lyr layer) SetW(weights *matrix.DenseMatrix) {
	ni := weights.Rows() / len(lyr)
	for i, node := range lyr {
		node.SetW(weights.GetMatrix(i*ni, 0, ni, 1))
	}
}

type node struct {
	fn      Function
	weights *matrix.DenseMatrix
}

func newNode(f Function, weights *matrix.DenseMatrix) *node {
	return &node{f, weights}
}
func (n *node) F(x *matrix.DenseMatrix) float64 {
	return n.fn.F(ensure(n.weights.Transpose().TimesDense(x)).Get(0, 0))
}
func (n *node) G(inputs *matrix.DenseMatrix, gradients []*matrix.DenseMatrix) *matrix.DenseMatrix {
	x := ensure(n.weights.Transpose().TimesDense(inputs)).Get(0, 0)
	dy := n.fn.G(x)
	ret := matrix.Zeros(1, 0)
	for i, gradient := range gradients {
		g := gradient.Copy()
		g.Scale(n.weights.Get(i, 0))
		ret = ensure(ret.Augment(g.Transpose()))
	}
	ret = ensure(ret.Augment(inputs.Transpose()))
	ret.Scale(dy)
	return ret.Transpose()
}
func (n *node) H(inputs *matrix.DenseMatrix, gradients, hessians []*matrix.DenseMatrix) *matrix.DenseMatrix {
	x := ensure(n.weights.Transpose().TimesDense(inputs)).Get(0, 0)
	ddy := n.fn.H(x)

	ni := len(gradients)
	li := 0
	if ni > 0 {
		li = gradients[0].Rows()
	}
	lw := n.weights.Rows()

	H := matrix.Zeros(ni*li+lw, ni*li+lw)

	for i, hessian := range hessians {
		hess := hessian.Copy()
		hess.Scale(n.weights.Get(i, 0))
		H.SetMatrix(i*li, i*li, hess)
	}

	for i, gradient := range gradients {
		H.SetMatrix(i*li, li*ni+i, gradient)
		H.SetMatrix(li*ni+i, i*li, gradient.Transpose())
	}

	xxt := ensure(inputs.TimesDense(inputs.Transpose()))
	xxt.Scale(ddy)
	H.SetMatrix(ni*li, ni*li, xxt)

	return H
}
func (n *node) W() *matrix.DenseMatrix {
	return n.weights
}
func (n *node) SetW(w *matrix.DenseMatrix) {
	n.weights = w
}

// The standard hyperbolic sigmoid activation function (tanh)
type HyperbolicSigmoid bool

func (h HyperbolicSigmoid) F(x float64) float64 {
	return math.Tanh(x)
}
func (h HyperbolicSigmoid) G(x float64) float64 {
	return 1 - math.Pow(math.Tanh(x), 2)
}
func (h HyperbolicSigmoid) H(x float64) float64 {
	y := math.Tanh(x)
	return 2 * (math.Pow(y, 3) - y)
}

// The standard hyperbolic sigmoid activation function (tanh)
type Identity bool

func (h Identity) F(x float64) float64 {
	return x
}
func (h Identity) G(x float64) float64 {
	return 1
}
func (h Identity) H(x float64) float64 {
	return 0 // For calculation purposes. This isn't actually true
}

func initWeights(n int, bound float64) *matrix.DenseMatrix {
	arr := make([]float64, n)
	for i := 0; i < n; i++ {
		arr[i] = (rand.Float64() - .5) * 2 * bound
	}
	return matrix.MakeDenseMatrix(arr, n, 1)
}

func ensure(A *matrix.DenseMatrix, err error) *matrix.DenseMatrix {
	if err != nil {
		panic(err)
	}
	return A
}
func ensureErr(err error) {
	if err != nil {
		panic(err)
	}
}
