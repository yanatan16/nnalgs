package nnalgs

import (
	"fmt"
	matrix "github.com/skelterjohn/go.matrix"
	"math"
	"math/rand"
)

var _ = fmt.Println

// A predictor
type Predictor interface {
	// Evaluate the predictor with predictor inputs as parameters and outputs as returns
	F(inputs []float64) float64
	// Evaluate the gradient of the predictor
	FG(inputs []float64) (float64, *matrix.DenseMatrix)
	// Evaluate the Hessian of the predictor
	FGH(inputs []float64) (float64, *matrix.DenseMatrix, *matrix.DenseMatrix)
	// Get the weight vector of this predictor
	W() *matrix.DenseMatrix
	// Set the weight vector for this predictor
	SetW(*matrix.DenseMatrix)
	// Length of the weight vector
	Len() int
}

// A trainer interface
type Trainer interface {
	// Train a predictor for a single round
	Train(p Predictor, seq DataSequence) Predictor
}

// A single sequence of temporal inputs and an outcome
type DataSequence struct {
	Xs [][]float64
	Z  float64
}

// A wrapper function around Trainer.Train() to repeat the training pattern
func Train(t Trainer, p Predictor, data DataStream) <-chan Predictor {
	c := make(chan Predictor)

	go func() {
		for {
			if seq, ok := <-data; ok {
				p = t.Train(p, seq)
				c <- p
			} else {
				break
			}
		}
		close(c)
	}()

	return c
}

type DataStream chan DataSequence

// Create an infinite iterator of a_k gain values in standard form
func StandardAk(a, A, alpha float64) chan float64 {
	c := make(chan float64)
	go func() {
		for k := 1; true; k++ {
			c <- a / math.Pow(float64(k)+A, alpha)
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


// Create an infinite iterator of Bernoulli +/- r i.i.d r.v.
func Bernoulli(r float64) chan float64 {
	c := make(chan float64)
	go func() {
		for {
			if rand.Float32() > .5 {
				c <- r
			} else {
				c <- -r
			}
		}
	}()
	return c
}

// Calculate the TD(lambda) function
// In general, TD(lambda) is not a gradient, just a root-finding function
func CalculateTDLFunc(predictor Predictor, seq DataSequence, lambda float64) *matrix.DenseMatrix {
	// Notes:
	// G is the gradient term accumulator
	// s is an accumulator of gradients of the predictor weighted by lambda
	//	That is, s_t+1 = lambda * s_t + dh/dtheta
	// f and g are the t+1'th evaluation of the predictor's evaluation and gradient
	// f1 and g1 represent the t'th evaluation of the predictor
	// So G += (f1 -f) * s_t

	n := predictor.Len()

	// Accumulate the gradient terms
	G := matrix.Zeros(n, 1)
	f1, g1 := predictor.FG(seq.Xs[0])
	f, g := f1, g1 // Initialize
	s := g1

	for i := 0; i < len(seq.Xs); i++ {
		if i < (len(seq.Xs) - 1) {
			// Evaluate the predictor
			f, g = predictor.FG(seq.Xs[i+1])
		} else {
			// The last round is the outcome
			f, g = seq.Z, matrix.Zeros(n, 1)
		}

		// Update dt
		sc := s.Copy()
		sc.Scale(f - f1)
		ensureErr(G.AddDense(sc))

		// Update s
		s.Scale(lambda)
		ensureErr(s.AddDense(g))

		// Save f/g
		f1, g1 = f, g
	}

	return G
}

// Calculate the TD(lambda) function and jacobian
// It is more efficient to caculate both together than separate due to
// the shared aggregation term s_t
// Since TD(lambda) is not a gradient in general, G is a root-finding function
//  and H is the Jacobian
func CalculateTDLFuncJacobian(predictor Predictor, seq DataSequence, lambda float64) (G, H *matrix.DenseMatrix) {
	// Notes:
	// G is the gradient term accumulator
	// H is the jacobian term accumulator
	// s is an accumulator of gradients of the predictor weighted by lambda
	//	That is, s_t+1 = lambda * s_t + dh/dtheta
	// sp is an accumulator of hessians of the predictor weighted by lambda
	//  That is, sp_t+1 = lambda * sp_t + d2h/dtheta2
	// f, g, and h are the t+1'th evaluation of the predictor's eval, gradient, and hessian
	// f, g, and h are the t'th evaluation of the predictor's eval, gradient, and hessian
	// So G += (f1 - f) * s_t
	// And H += (g1 - g) * s_t^T + (f1 - f) * sp_t

	n := predictor.Len()
	G = matrix.Zeros(n, 1) // Accumulate the gradient terms
	H = matrix.Zeros(n, n) // Accumulate the jacobian terms

	f1, g1, h1 := predictor.FGH(seq.Xs[0])
	f, g, h := f1, g1, h1
	s := g1
	sp := h1

	for i := 0; i < len(seq.Xs); i++ {
		if i < (len(seq.Xs) - 1) {
			// Evaluate the predictor
			f, g, h = predictor.FGH(seq.Xs[i+1])
		} else {
			// For the last round, f = Z, g/h = 0
			f, g, h = seq.Z, matrix.Zeros(n, 1), matrix.Zeros(n, n)
		}

		// Update Gradient
		sc := s.Copy()
		sc.Scale(f - f1)
		ensureErr(G.AddDense(sc))

		// Update Jacobian part 1
		gc := g.Copy()
		ensureErr(gc.SubtractDense(g1))
		ensureErr(H.AddDense(ensure(gc.TimesDense(s.Transpose()))))

		// Jacobian part 2
		spc := sp.Copy()
		spc.Scale(f - f1)
		ensureErr(H.AddDense(spc))

		// Update s
		s.Scale(lambda)
		ensureErr(s.AddDense(g))

		// Update sprime
		sp.Scale(lambda)
		ensureErr(sp.AddDense(h))

		// Save f/g/h
		f1, g1, h1 = f, g, h
	}

	return G, H
}

// Trainer for the TD(lambda) trainer
type TDLambda struct {
	Lambda float64
	Ak     chan float64
}

func NewTDLambda(lambda float64, ak chan float64) *TDLambda {
	return &TDLambda{
		Lambda: lambda,
		Ak:     ak,
	}
}

// Perform TD(lambda)
func (tdl *TDLambda) Train(predictor Predictor, seq DataSequence) Predictor {
	G := CalculateTDLFunc(predictor, seq, tdl.Lambda)
	G.Scale(<-tdl.Ak)
	G.Scale(-1)

	theta := predictor.W()
	ensureErr(theta.SubtractDense(G))
	predictor.SetW(theta)

	return predictor
}

// Trainer for the Second Order extension of the TD(lambda) trainer
type SecondOrderTDLambda struct {
	Lambda     float64
	Ak, Deltak chan float64
}

func NewSecondOrderTDLambda(lambda float64, ak, dk chan float64) *SecondOrderTDLambda {
	return &SecondOrderTDLambda{
		Lambda: lambda,
		Ak:     ak,
		Deltak: dk,
	}
}

// Perform the second-order version of TD(lambda)
func (tdl *SecondOrderTDLambda) Train(predictor Predictor, seq DataSequence) Predictor {
	G, H := CalculateTDLFuncJacobian(predictor, seq, tdl.Lambda)

	// Ensure invertable H
	mod := matrix.Eye(H.Rows())
	mod.Scale(<-tdl.Deltak)
	ensureErr(H.AddDense(mod))

	dt := ensure(H.SolveDense(G))
	dt.Scale(<-tdl.Ak)

	theta := predictor.W()
	ensureErr(theta.SubtractDense(dt))

	if math.IsNaN(theta.OneNorm()) {
		panic("NaN in Prediction")
	}

	predictor.SetW(theta)

	return predictor
}

// Return a perturbation vector
func Perturb(n int, perts chan float64) *matrix.DenseMatrix {
	delta := make([]float64, n)
	for i := 0; i < n; i++ {
		delta[i] = <-perts
	}
	return matrix.MakeDenseMatrix(delta, n, 1)
}

// The enhanced adaptive second order form of TD(lambda)
// References: Spall 2003, 2009
// Perts should an iterator if iid r.v.'s for Perturbation vectors
type AdaptiveTDLambda struct {
	Lambda                float64
	Ak, Ck, Deltak, Perts chan float64
	hbar, hbarbar         *matrix.DenseMatrix
	k                     int
	csum                  float64
	Block                 bool
}

func NewAdaptiveTDLambda(predictor Predictor, lambda float64, ak, ck, dk chan float64) *AdaptiveTDLambda {
	n := predictor.Len()
	return &AdaptiveTDLambda{
		Lambda:  lambda,
		Ak:      ak,
		Ck:      ck,
		Deltak:  dk,
		Perts:   Bernoulli(1),
		hbar:    matrix.Zeros(n, n),
		hbarbar: matrix.Zeros(n, n),
		csum:    0,
		Block:   true,
	}
}

func ElementInversion(t *matrix.DenseMatrix) *matrix.DenseMatrix {
	s := matrix.Zeros(t.Rows(), t.Cols())
	for i := 0; i < t.Rows(); i++ {
		for j := 0; j < t.Cols(); j++ {
			s.Set(i, j, 1.0/t.Get(i, j))
		}
	}
	return s
}

// Perform the approximated second-order version of TD(lambda) (Spall / Sutton)
func (tdl *AdaptiveTDLambda) Train(predictor Predictor, seq DataSequence) Predictor {
	theta := predictor.W()

	Er := PredictionError(predictor, seq)
	G := CalculateTDLFunc(predictor, seq, tdl.Lambda)
	H := tdl.EstimateJacobian(predictor, seq, theta)

	dt := ensure(H.SolveDense(G))
	dt.Scale(<-tdl.Ak)

	thetaNew := theta.Copy()
	ensureErr(thetaNew.SubtractDense(dt))
	predictor.SetW(thetaNew)

	if ErNew := PredictionError(predictor, seq); ErNew > Er && tdl.Block {
		// Repeal Change
		predictor.SetW(theta)
	} else if math.IsNaN(ErNew) {
		fmt.Println(<-tdl.Deltak)
		fmt.Println(H)
		fmt.Println(H.Solve(G))
		panic("NaN in Prediction")
	}

	return predictor
}

// Estimate the Jacobian from the Adaptive TD(lambda)
// The enhanced version has a feedback term and a weighting term relative to ck
func (tdl *AdaptiveTDLambda) EstimateJacobian(predictor Predictor, seq DataSequence, theta *matrix.DenseMatrix) *matrix.DenseMatrix {
	n := predictor.Len()

	// Get delta vector
	ck := <-tdl.Ck
	delta := Perturb(n, tdl.Perts)
	scaledDelta := delta.Copy()
	scaledDelta.Scale(ck)

	// Evaluate theta + ck * delta
	tp1 := theta.Copy()
	ensureErr(tp1.AddDense(scaledDelta))
	predictor.SetW(tp1)
	Gp1 := CalculateTDLFunc(predictor, seq, tdl.Lambda)

	// Evaluate theta - ck * delta
	tn1 := theta.Copy()
	ensureErr(tn1.SubtractDense(scaledDelta))
	predictor.SetW(tn1)
	Gn1 := CalculateTDLFunc(predictor, seq, tdl.Lambda)

	// Calculate the hessian/jacobian estimate
	dG := Gp1
	ensureErr(dG.SubtractDense(Gn1))
	scaledDelta.Scale(2)
	iScaledDelta := ElementInversion(scaledDelta)

	// Because Hest is a Jacobian, this is the form
	Hest := ensure(dG.TimesDense(iScaledDelta.Transpose()))

	// Find the feedback term
	Dk := ensure(delta.TimesDense(ElementInversion(delta).Transpose()))
	ensureErr(Dk.SubtractDense(matrix.Eye(n)))
	Psi := ensure(tdl.hbarbar.TimesDense(Dk))
	ensureErr(Hest.SubtractDense(Psi))

	// Find the weighting term
	ck2 := math.Pow(ck, 2)
	tdl.csum += ck2
	wk := ck2 / tdl.csum

	// Update Hbar by weighting previous estimate with new estimate
	tdl.hbar.Scale(1 - wk)
	Hest.Scale(wk)
	ensureErr(tdl.hbar.AddDense(Hest))

	// Map hbar to hbarbar to ensure Invertibility
	tdl.hbarbar = tdl.hbar.Copy()
	mod := matrix.Eye(n)
	mod.Scale(<-tdl.Deltak)
	tdl.hbarbar.AddDense(mod)

	return tdl.hbarbar
}
