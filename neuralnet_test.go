// nnalgs provides an simple Neural Network library with access to gradients and Hessians.
package nnalgs

import (
	matrix "github.com/skelterjohn/go.matrix"
	"math"
	"testing"
)

func TestFullNetwork(t *testing.T) {
	nhidden := 3
	ninputs := 2
	weights := []float64{-1, -1, -1, // Node 1
		0, 0, 0, // Node 2
		1, 1, 1, // Node 3
		.1, .2, .3, .4, // linear layer
	}
	x := []float64{-.5, .5}

	expOutput := .55231

	expGradient := matrix.MakeDenseMatrix([]float64{
		.1 * .41997 * -.5, // theta_21 * f'(theta1 * x^) x^
		.1 * .41997 * .5,
		.1 * .41997 * 1,
		.2 * 1 * -.5, // theta_22 * f'(theta2 * x^) x^
		.2 * 1 * .5,
		.2 * 1 * 1,
		.3 * .41997 * -.5, // theta_23 * f'(theta3 * x^) x^
		.3 * .41997 * .5,
		.3 * .41997 * 1,
		-.76159, // f(theta1 * x^)
		0,       // f(theta2 * x^)
		.76159,  // f(theta3 * x^)
		1,       // 1
	}, len(weights), 1)

	expHessian := matrix.MakeDenseMatrix([]float64{
		.063970 * .25, .063970 * -.25, .063970 * -.5, 0, 0, 0, 0, 0, 0, .41997 * -.5, 0, 0, 0, // theta_21 * f''(theta1 * x^) x^T * x^ | 0 | 0 | f'(theta1 * x^) x^ | 0 | 0 | 0
		.063970 * -.25, .063970 * .25, .063970 * .5, 0, 0, 0, 0, 0, 0, .41997 * .5, 0, 0, 0,
		.063970 * -.5, .063970 * .5, .063970, 0, 0, 0, 0, 0, 0, .41997, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -.5, 0, 0, // theta_21 * f''(theta1 * x^) x^T * x^ | 0 | 0 | f'(theta1 * x^) x^ | 0 | 0 | 0
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .5, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 3 * -.063970 * .25, 3 * -.063970 * -.25, 3 * -.063970 * -.5, 0, 0, .41997 * -.5, 0, // theta_21 * f''(theta1 * x^) x^T * x^ | 0 | 0 | f'(theta1 * x^) x^ | 0 | 0 | 0
		0, 0, 0, 0, 0, 0, 3 * -.063970 * -.25, 3 * -.063970 * .25, 3 * -.063970 * .5, 0, 0, .41997 * .5, 0,
		0, 0, 0, 0, 0, 0, 3 * -.063970 * -.5, 3 * -.063970 * .5, 3 * -.063970, 0, 0, .41997, 0,
		.41997 * -.5, .41997 * .5, .41997, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, -.5, .5, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, .41997 * -.5, .41997 * .5, .41997, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}, len(weights), len(weights))

	wmat := matrix.MakeDenseMatrixStacked([][]float64{weights}).Transpose()

	network := NewNetwork(ninputs, []int{nhidden, 1}, HyperbolicSigmoid(true), 2)
	network.SetW(wmat)

	output, gradient, hessian := network.FGH(x)

	if !matrix.Equals(network.W(), wmat) {
		t.Error("Weights not set correctly", network.W())
	}
	if network.Len() != len(weights) {
		t.Error("weights length incorrect!", len(weights), network.Len())
	}
	if math.Abs(output-expOutput) > .001 {
		t.Error("Expected output incorrect", expOutput, network.F(x))
	}
	if !matrix.ApproxEquals(gradient, expGradient, .001) {
		t.Error("Expected gradient incorrect", ensure(expGradient.Augment(gradient)))
	}
	if !matrix.ApproxEquals(hessian, expHessian, .001) {
		t.Log(expHessian)
		t.Log(hessian)
		expHessian.SubtractDense(hessian)
		t.Error("Expected hessian incorrect", expHessian)
	}
}

func TestLinearNetwork(t *testing.T) {
	network := NewNetwork(4, []int{1}, HyperbolicSigmoid(true), 1)
	w := network.W()
	n := network.Len()
	_, g, h := network.FGH([]float64{1, 2, 3, 4})

	if n != w.Rows() || n != 5 {
		t.Error("Weighting size is wrong")
	}
	if !matrix.Equals(g, matrix.MakeDenseMatrixStacked([][]float64{{1, 2, 3, 4, 1}}).Transpose()) {
		t.Error("Gradient is incorrect!")
	}
	if !matrix.Equals(h, matrix.Zeros(n, n)) {
		t.Error("Hessian is incorrect!")
	}
}

func TestElementInversion(t *testing.T) {
	x := matrix.MakeDenseMatrix([]float64{1, 2, 3, 4, 5}, 5, 1)
	y := ElementInversion(x)
	if !matrix.Equals(y, matrix.MakeDenseMatrix([]float64{1, .5, 1.0 / 3, .25, .2}, 5, 1)) {
		t.Fail()
	}
}

func TestEquivalentOutcomes(t *testing.T) {
	network := NewNetwork(4, []int{5, 1}, HyperbolicSigmoid(true), 2)
	f1, g1, _ := network.FGH([]float64{1, 2, 3, 4})
	f2, g2 := network.FG([]float64{1, 2, 3, 4})
	f3 := network.F([]float64{1, 2, 3, 4})

	if f1 != f2 || f2 != f3 {
		t.Error("Function evals are different between F/G/H")
	}
	if !matrix.ApproxEquals(g1, g2, .00001) {
		t.Error("Function gradients are different between G/H")
	}
}

func TestHessian(t *testing.T) {
	network := NewNetwork(4, []int{5, 1}, HyperbolicSigmoid(true), 2)
	_, g, h := network.FGH([]float64{1, 2, 3, 4})
	if g.Rows() != h.Rows() {
		t.Error("rows don't match up")
	}

	H := h
	if !matrix.Equals(H, H.Transpose()) {
		t.Fail()
	}
	H2 := ensure(H.Transpose().TimesDense(H))
	mod := matrix.Eye(H2.Rows())
	mod.Scale(.1)
	_ = H2.AddDense(mod)
	if _, err := H2.Cholesky(); err != nil {
		t.Error("Cannot use cholesky on HH", err)
	}
}

func TestLayer(t *testing.T) {
	layer := newLayer(4, 3, HyperbolicSigmoid(true), 2)
	w := matrix.MakeDenseMatrix(make([]float64, 4*4), 4*4, 1)
	layer.SetW(w)
	if w2 := layer.W(); !matrix.Equals(w, w2) {
		t.Error("Setting layer weights failed", w.Rows(), w2.Rows())
	}
}

func TestHyperbolicSigmoidNode(t *testing.T) {
	node := newNode(HyperbolicSigmoid(true), initWeights(5, 2))

	verifyWeights(t, node.W(), 5, 2)
	w2 := matrix.MakeDenseMatrix([]float64{1, -2, 3, -4, 5}, 5, 1)
	node.SetW(w2)
	if !matrix.Equals(w2, node.W()) {
		t.Error("Setting weights failed", node.W())
	}

	x := matrix.MakeDenseMatrix([]float64{5, 4, 3, 2, 1}, 5, 1)
	if math.Abs(node.F(x)-.99505) > .001 {
		t.Error("Linear evaluation failed", node.F(x))
	}

	emptyg := make([]*matrix.DenseMatrix, 0)
	grad := node.G(x, emptyg) // Input gradients
	expGrad1 := x.Copy()
	expGrad1.Scale(.00986694)
	if !matrix.ApproxEquals(expGrad1, grad, .001) {
		t.Error("Gradient with no other input gradients incorrect", ensure(expGrad1.Augment(grad)))
	}

	xxt := ensure(x.TimesDense(x.Transpose()))
	hess := node.H(x, emptyg, emptyg)
	expHess := xxt.Copy()
	expHess.Scale(-.0196345)
	if !matrix.ApproxEquals(expHess, hess, .001) {
		t.Error("Hessian with no inputs incorrect", expHess, hess)
	}
}

func TestLinearNode(t *testing.T) {
	node := newNode(Identity(true), initWeights(5, 2))

	verifyWeights(t, node.W(), 5, 2)
	w2 := matrix.MakeDenseMatrix([]float64{1, 2, 3, 4, 5}, 5, 1)
	node.SetW(w2)
	if !matrix.Equals(w2, node.W()) {
		t.Error("Setting weights failed", node.W())
	}

	x := matrix.MakeDenseMatrix([]float64{5, 4, 3, 2, 1}, 5, 1)
	if node.F(x) != 35 {
		t.Error("Linear evaluation failed", node.F(x))
	}

	emptyg := make([]*matrix.DenseMatrix, 0)
	grad := node.G(x, emptyg) // Input gradients
	if !matrix.Equals(x, grad) {
		t.Error("Gradient with no other input gradients incorrect", grad)
	}

	g := []*matrix.DenseMatrix{
		matrix.Ones(1, 1),
		matrix.Ones(1, 1),
		matrix.Ones(1, 1),
		matrix.Ones(1, 1),
	}
	grad = node.G(x, g)
	expGrad := matrix.MakeDenseMatrix([]float64{
		1, 2, 3, 4, 5, 4, 3, 2, 1,
	}, 9, 1)
	if !matrix.Equals(expGrad, grad) {
		t.Error("Gradient with input gradients incorrect", grad)
	}

	hess := node.H(x, emptyg, emptyg)
	expHess := matrix.Zeros(5, 5)
	if !matrix.Equals(expHess, hess) {
		t.Error("Hessian with no inputs incorrect", hess)
	}

	hess = node.H(x, g, g)
	expHess = matrix.Zeros(9, 9)
	for i := 0; i < 4; i++ {
		expHess.Set(i, i, 1*w2.Get(i, 0))
		expHess.Set(i, 4+i, 1)
		expHess.Set(4+i, i, 1)
	}
	if !matrix.Equals(expHess, hess) {
		t.Error("Hessian with inputs incorrect", hess, expHess)
	}
}

func TestMakeWeights(t *testing.T) {
	weights := initWeights(5, 2)
	verifyWeights(t, weights, 5, 2)
}

func verifyWeights(t *testing.T, weights *matrix.DenseMatrix, n int, b float64) {
	if weights.Rows() != n || weights.Cols() != 1 {
		t.Error("Lenght of weights incorrect", weights)
	}
	max, min := weights.Get(0, 0), weights.Get(0, 0)
	for _, ww := range weights.Array()[1:] {
		max = math.Max(max, ww)
		min = math.Max(min, ww)
	}
	if max > b || min < -b {
		t.Error("max or min is incorrect", max, min)
	}
}

func TestIdentity(t *testing.T) {
	f := Identity(true)
	if f.F(-1000) != -1000 || f.F(0) != 0 || f.F(1000) != 1000 {
		t.Error("Identity incorrect")
	}
	if f.G(-1000) != 1 || f.G(0) != 1 || f.G(1000) != 1 {
		t.Error("Identity derivative incorrect")
	}
	if f.H(-1000) != 0 || f.H(0) != 0 || f.H(1000) != 0 {
		t.Error("Identity hessian incorrect")
	}
}

func TestHyperbolicSigmoid(t *testing.T) {
	f := HyperbolicSigmoid(true)
	exerciseSigmoidFunction(t, f)
}

func exerciseSigmoidFunction(t *testing.T, f Function) {
	if f.F(-1000) > -.99 || f.F(-1000) < -1.0 {
		t.Error("Negative asymptote incorrect: ", f.F(-1000))
	}
	if f.F(1000) < .99 || f.F(1000) > 1.0 {
		t.Error("Positive asymptote incorrect", f.F(1000))
	}
	if f.F(0) != 0 || f.F(1) < 0 || f.F(-1) > 0 {
		t.Error("Around zero incorrect", f.F(-1), f.F(0), f.F(1))
	}

	if f.G(1) < 0 || f.G(1000) > .01 || f.G(1000) < 0 {
		t.Error("Positive gradient incorrect", f.G(1000))
	}
	if f.G(-1) < 0 || f.G(-1000) > .01 || f.G(-1000) < 0 {
		t.Error("Negative gradient incorrect", f.G(-1), f.G(-1000))
	}
	if f.G(0) != 1 {
		t.Error("Inflection point gradient incorrect", f.G(0))
	}

	if f.H(0) != 0 {
		t.Error("Inflection point hessian incorrect", f.H(0))
	}
	if f.H(1) > 0 {
		t.Error("Postive bend incorrect", f.H(1))
	}
	if f.H(-1) < 0 {
		t.Error("Negative bend incorrect", f.H(-1))
	}
}
