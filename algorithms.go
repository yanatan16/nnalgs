package nnalgs

import (
	"math"
	"fmt"
	matrix "github.com/skelterjohn/go.matrix"
)
var _ = fmt.Println

type Pair struct {
	X []float64
	Z float64
}

// Create an infinite iterator of a_k gain values in standard form
func StandardAk(a, A, alpha float64) chan float64 {
	c := make(chan float64)
	go func () {
		for k := 1; true; k++ {
			c <- a / math.Pow(float64(k) + A, alpha)
		}
	}()
	return c
}

// Create an infinite iterator of c_k gain values in standard form
func StandardCk(c, gamma float64) chan float64 {
	ch := make(chan float64)
	go func() {
		for k := 1; true; k++ {
			ch <- c / math.Pow(float64(k), gamma)
		}
	}()
	return ch
}

// Create a data channel based on a set already in memory and send it through n rounds
func Dataset(set []Pair, rounds int) chan Pair {
	c := make(chan Pair)
	go func () {
		for i := 0; i < rounds; i++ {
			for _, p := range set {
				c <- p
			}
		}
		close(c)
	}()
	return c
}

// Calculate prediction errors
// Returns mean error, mean squared error, and standard deviation
func PredictionErrors(network Network, data chan Pair, n int) (mu, mse, sigma float64) {
	var sum, sumsq float64

	agg := make([]float64, 0, n)
	i := 0
	for pair := range data {

		x := network.F(pair.X)[0] - pair.Z
		sum += x
		sumsq += math.Pow(x, 2)

		agg = append(agg, x)

		i += 1
		if i >= n {
			break
		}
	}
	mean, meansq := sum / float64(i), sum / float64(i)

	var sumerrsq float64
	for _, x := range agg {
		sumerrsq += math.Pow(x - mean, 2)
	}
	variance := sumerrsq / float64(len(agg) - 1)

	return mean, meansq, math.Sqrt(variance)
}

// Perform Backpropogation
func Backpropogation(network Network, data chan Pair, achan chan float64, stop int) {
	theta := network.W()
	k := 0
	for pair := range data {
		ak := <- achan
		f, g := network.FG(pair.X)
		dt := g[0]
		dt.Scale(ak * (f[0] - pair.Z))

		ensureErr(theta.SubtractDense(dt))
		network.SetW(theta)

		k += 1
		if k >= stop {
			break
		}
	}
}

// Perform the second-derivative version of backpropogation
func NewtonRaphsonBackpropogation(network Network, data chan Pair, stop int) {
	theta := network.W()
	mod := matrix.Eye(theta.Rows())
	mod.Scale(1) // To ensure invertability
	k := 0
	for pair := range data {
		f, g, h := network.FGH(pair.X)

		h[0].AddDense(mod)
		dt := ensure(ensure(h[0].Inverse()).TimesDense(g[0]))
		dt.Scale(f[0] - pair.Z)

		ensureErr(theta.SubtractDense(dt))
		network.SetW(theta)

		k += 1
		if k >= stop {
			break
		}
	}
}

// Return a perturbation vector
func Perturb(n int, perts chan float64) *matrix.DenseMatrix {
	delta := make([]float64, n)
	for i := 0; i < n; i++ {
		delta[i] = <- perts
	}
	return matrix.MakeDenseMatrix(delta, n, 1)
}

// Estimate the Hessian from the gradient simultaneous perturbation
func EstimateH(nework Network, data []float64, theta, delta *matrix.DenseMatrix) *matrix.DenseMatrix {
	delta := Perturb(theta.Rows(), theta)
	delta2 := delta.Copy()

	delta2.Scale(ck)
	ensureErr(theta.AddDense(delta2));
	f1, g1 := network.FG(pair.X)

	delta2.Scale(2)
	ensureErr(theta.SubtractDense(delta2));
	f2, g2 := network.FG(pair.X)

	dg := g1.Copy()
	dg.SubtractDense(g2)
	for i := 0; i < n; i++ {
		delta.Set(i, 0, 1.0 / delta.Get(i, 0))
	}
	dgt := ensure(dg.Transpose().TimesDense(delta));
	dgt.Scale(1.0 / (2 * ck))

	Hhat := dgt.Copy()
	Hhat.AddDense(dgt.Transpose())
	Hhat.Scale(.5)

	return Hhat
}

// Square root a matrix using the positive definite form
func SquareRoot(A *matrix.DenseMatrix) error {
	L, err := A.Cholesky()
	D := matrix.Diagnol(L.DiagnolCopy())
	L.FillDiagnol(matrix.Ones(A.Rows(), 1).Array())

	// Now we have L^T * D * L = A
}

// Map a hessian estimate into an invertible form
func EnsureInvertible(Hest *matrix.DenseMatrix) {
	Hbb := Hest.Copy()
	Hbb = ensure(Hbb.TimesDense(Hbb))
}

// Perform the approximated second-derivative version of backpropogation (Spall 2000)
// perts should be a channel of perturbation vector elements, randomly selected and iid
func AdaptiveBackpropogation(network Network, data chan Pair, achan, cchan, perts chan float64, stop int) {
	theta := network.W()
	n := theta.Rows()

	mod := matrix.Eye(theta.Rows())
	mod.Scale(1) // To ensure invertability
	Hbar := matrix.Eye(theta.Rows())

	k := 0
	for pair := range data {
		ak, ck := <- achan, <- cchan

		f0, g0 := network.FG(pair.X)
		Hhat := EstimateH(network, pair.X, thtea, perts)

		HHat.Scale(1.0 / (k+1))
		Hbar.Scale(k / (k+1))
		ensureErr(Hbar.AddDense(Hhat))




		h[0].AddDense(mod)
		dt := ensure(ensure(h[0].Inverse()).TimesDense(g[0]))
		dt.Scale(f[0] - pair.Z)

		ensureErr(theta.SubtractDense(dt))
		network.SetW(theta)

		k += 1
		if k >= stop {
			break
		}
	}
}