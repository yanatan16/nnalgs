package nnalgs

import (
	"fmt"
	matrix "github.com/skelterjohn/go.matrix"
	"math"
	"math/rand"
	"testing"
)

var nt, nv int = 10000, 1000
var nt1, nt2 int = 1000, 9000

func Xor(a, b float64) float64 {
	if a == b {
		return 0
	}
	return 1
}

func SimpleQuadratic(a, b float64) float64 {
	return .2*a*a + b*b + .3*a*b - a - b + .4
}

func RandomStep(x float64) float64 {
	return x + rand.Float64()
}

func RandomWalkDataStream(n int) DataStream {
	ch := make(DataStream)
	go func() {
		for {
			m := (int(rand.Int31()) % n) + 1
			x := make([][]float64, m)
			var z float64
			for i := 0; i < m; i++ {
				z = RandomStep(z)
				x[i] = []float64{float64(m - i), z}
			}
			z = RandomStep(z)

			seq := DataSequence{Xs: x, Z: z}
			ch <- seq
		}
	}()
	return ch
}

func XorDataStream() DataStream {
	ch := make(DataStream)
	go func() {
		for {
			a, b := math.Floor(rand.Float64()+.5), math.Floor(rand.Float64()+.5)
			c := Xor(a, b)
			p := DataSequence{Xs: [][]float64{{a, b}}, Z: c}
			ch <- p
		}
	}()
	return ch
}

func QuadDataStream() DataStream {
	ch := make(DataStream)
	go func() {
		for {
			a, b := rand.Float64()*2-1, rand.Float64()*2-1
			c := SimpleQuadratic(a, b)
			p := DataSequence{Xs: [][]float64{{a, b}}, Z: c}
			ch <- p
		}
	}()
	return ch
}

func RunTrainer(t *testing.T, trainer Trainer, pred Predictor, data DataStream, nt, nv int) float64 {
	premse := PredictionErrors(pred, data, nv)
	preds := Train(trainer, pred, data)

	for i := 0; i < nt-1; i++ {
		<-preds
	}

	final := <-preds
	postmse := PredictionErrors(final, data, nv)

	if premse <= postmse || math.IsNaN(postmse) {
		t.Error("Trainer failed to lower MSE: pre", premse, "post", postmse)
	} else {
		t.Log("Trainer lowered MSE: pre", premse, "post", postmse)
	}
	return postmse
}

func TestTrainXor(t *testing.T) {
	network := NewNetwork(2, []int{2, 1}, HyperbolicSigmoid(true), 1)
	ak := StandardAk(1, float64(nt/10), .501)
	data := XorDataStream()

	trainer := NewTDLambda(0, ak)

	mse := RunTrainer(t, trainer, network, data, nt, nv)

	fmt.Println("Xor TD(0) mse", mse)
}

func TestTrainXorLinear(t *testing.T) {
	network := NewNetwork(2, []int{1}, HyperbolicSigmoid(true), 1)
	ak := StandardAk(1, float64(nt/10), .501)
	data := XorDataStream()

	trainer := NewTDLambda(0, ak)

	mse := RunTrainer(t, trainer, network, data, nt, nv)

	fmt.Println("Linear Xor TD(0) mse", mse)
}

func TestTrainQuad(t *testing.T) {
	network := NewNetwork(2, []int{5, 1}, HyperbolicSigmoid(true), 1)
	ak := StandardAk(10, float64(nt/10), .501)
	data := QuadDataStream()

	trainer := NewTDLambda(0, ak)

	mse := RunTrainer(t, trainer, network, data, nt, nv)

	fmt.Println("Quad TD(0) mse", mse)
	// fmt.Println(SprintSeq(network, <- data))
}

func TestTrainQuadLinear(t *testing.T) {
	network := NewNetwork(2, []int{1}, HyperbolicSigmoid(true), 1)
	ak := StandardAk(10, float64(nt/10), .501)
	data := QuadDataStream()

	trainer := NewTDLambda(0, ak)

	mse := RunTrainer(t, trainer, network, data, nt, nv)

	fmt.Println("Linear Quad TD(0) mse", mse)
	// fmt.Println(SprintSeq(network, <- data))
}

func TestTrainQuadNewton(t *testing.T) {
	network := NewNetwork(2, []int{5, 1}, HyperbolicSigmoid(true), 1)
	ak := StandardAk(1, float64(nt1/10), .501)
	data := QuadDataStream()

	trainer := NewTDLambda(0, ak)
	mse := RunTrainer(t, trainer, network, data, nt1, nv)

	ak = StandardAk(1, float64(nt2/10), .501)
	dk := StandardCk(.01, 2)
	trainer2 := NewSecondOrderTDLambda(0, ak, dk)

	mse2 := RunTrainer(t, trainer2, network, data, nt2, nv)

	fmt.Println("Quad TD(0) mse", mse, " 2O-TD(0) mse", mse2)
	// fmt.Println(SprintSeq(network, <- data))
}

func TestTrainQuadAdaptive(t *testing.T) {
	network := NewNetwork(2, []int{5, 1}, HyperbolicSigmoid(true), 1)
	ak := StandardAk(1, float64(nt1/10), .501)
	data := QuadDataStream()

	trainer := NewTDLambda(0, ak)
	mse := RunTrainer(t, trainer, network, data, nt1, nv)

	ak = StandardAk(.01, float64(nt2/10), .602)
	ck := StandardCk(.01, .101)
	dk := StandardCk(.01, 1)
	trainer2 := NewAdaptiveTDLambda(network, 0, ak, ck, dk)

	mse2 := RunTrainer(t, trainer2, network, data, nt2, nv)

	fmt.Println("Quad TD(0) mse", mse, " Adap-TD(0) mse", mse2)
	// fmt.Println(SprintSeq(network, <- data))
}

// func TestTrainQuadCompareAdaptive2Order(t *testing.T) {
// 	network := NewNetwork(2, []int{ 1}, HyperbolicSigmoid(true), 1)
// 	ak := StandardAk(1, float64(nt1/10), .501)
// 	data := QuadDataStream()

// 	trainer := NewTDLambda(0, ak)
// 	RunTrainer(t, trainer, network, data, nt1, nv)

// 	ak = StandardAk(1, float64(nt2/10), .602)
// 	ck := StandardCk(.001, .101)
// 	dk := StandardCk(0, 1)
// 	trainer1 := NewAdaptiveTDLambda(network, 0, ak, ck, dk)

// 	seq := <- data
// 	G, H := CalculateTDLFuncJacobian(network, seq, 0)

// 	theta := network.W()
// 	H2 := matrix.Zeros(H.Rows(), H.Rows())
// 	for i := 0; i < 20; i++ {
// 		ensureErr(H2.AddDense(trainer1.EstimateJacobian(network, seq, theta)))
// 	}
// 	H2.Scale(.05)

// 	Hp := H.Copy()
// 	ensureErr(Hp.SubtractDense(H2))

// 	fmt.Println(SprintSeq(network, seq))
// 	fmt.Println("Hessian estimates difference:")
// 	fmt.Println(ensure(G.TimesDense(G.Transpose())))
// 	fmt.Println(H)
// 	fmt.Println(H2)
// 	fmt.Println(Hp)
// }

func TestTrainRandomWalk(t *testing.T) {
	network := NewNetwork(2, []int{4, 1}, HyperbolicSigmoid(true), 1)
	ak := StandardAk(1, float64(nt/10), .501)
	data := RandomWalkDataStream(10)

	trainer := NewTDLambda(0, ak)

	mse := RunTrainer(t, trainer, network, data, nt, nv)

	fmt.Println("RandomWalk TD(1) mse", mse)
	// fmt.Println(SprintSeq(network, <- data))
}

func TestTrainRandomWalkLinear(t *testing.T) {
	network := NewNetwork(2, []int{1}, HyperbolicSigmoid(true), 1)
	ak := StandardAk(1, float64(nt/10), .501)
	data := RandomWalkDataStream(10)

	trainer := NewTDLambda(0, ak)

	mse := RunTrainer(t, trainer, network, data, nt, nv)

	fmt.Println("Linear RandomWalk TD(1) mse", mse)
}

func TestTrainRandomWalkNewton(t *testing.T) {
	network := NewNetwork(2, []int{4, 1}, HyperbolicSigmoid(true), 1)
	ak := StandardAk(1, float64(nt1/10), .501)
	data := RandomWalkDataStream(10)

	trainer := NewTDLambda(0, ak)
	mse := RunTrainer(t, trainer, network, data, nt1, nv)

	ak = StandardAk(.01, float64(nt2/10), 1)
	dk := StandardCk(.1, 2)
	trainer2 := NewSecondOrderTDLambda(.5, ak, dk)

	mse2 := RunTrainer(t, trainer2, network, data, nt2, nv)

	fmt.Println("RandomWalk TD(1) mse", mse, " 2O-TD(1) mse", mse2)
}

func TestTrainRandomWalkAdaptive(t *testing.T) {
	network := NewNetwork(2, []int{4, 1}, HyperbolicSigmoid(true), 1)
	ak := StandardAk(1, float64(nt1/10), .501)
	data := RandomWalkDataStream(10)

	trainer := NewTDLambda(0, ak)
	mse := RunTrainer(t, trainer, network, data, nt1, nv)

	ak = StandardAk(.1, float64(nt2/10), .602)
	ck := StandardCk(.01, .101)
	dk := StandardCk(.001, 2)
	trainer2 := NewAdaptiveTDLambda(network, .3, ak, ck, dk)

	mse2 := RunTrainer(t, trainer2, network, data, nt2, nv)

	fmt.Println("RandomWalk TD(1) mse", mse, " Adap-TD(1) mse", mse2)
}

func TestGradientsAndHessians(t *testing.T) {
	network := NewNetwork(2, []int{2, 1}, HyperbolicSigmoid(true), 1)
	ak := StandardAk(1, float64(nt1/10), .501)
	data := RandomWalkDataStream(10)
	seq := <-data

	ak = StandardAk(1, float64(nt2/10), .602)
	ck := StandardCk(.1, .101)
	dk := StandardCk(1, 1)
	trainer2 := NewAdaptiveTDLambda(network, 1, ak, ck, dk)

	G1, H := CalculateTDLFuncJacobian(network, seq, 0)
	if G2 := CalculateTDLFunc(network, seq, 0); !matrix.ApproxEquals(G2, G1, .00001) {
		G3 := G1.Copy()
		ensureErr(G3.SubtractDense(G2))
		t.Error("Gradient calculations are off!")
		t.Log(G1)
		t.Log(G2)
		t.Log(G3)
	}

	H2 := trainer2.EstimateJacobian(network, seq, network.W())
	if H.Rows() != H.Cols() || H.Rows() != G1.Rows() ||
		H.Rows() != network.W().Rows() || H.Rows() != network.Len() ||
		H.Rows() != H2.Rows() || H2.Rows() != H2.Cols() {
		t.Error("Some dimensions are wrong")
	}
}
