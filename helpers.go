package nnalgs

import (
	"fmt"
	"math"
)

// Create a data channel based on a set already in memory and send it through n rounds
func Dataset(set []DataSequence, rounds int) DataStream {
	c := make(DataStream)
	go func() {
		for i := 0; i < rounds; i++ {
			for _, p := range set {
				c <- p
			}
		}
		close(c)
	}()
	return c
}

type ValidationSet struct {
	set []DataSequence
	begin, end int
}

func (t ValidationSet) Test() DataStream {
	return Dataset(t.set[t.begin:t.end], 1)
}
func (t ValidationSet) Train(rounds int) DataStream {
	c := make(DataStream)
	go func() {
		for i := 0; i < rounds; i++ {
			for _, p := range t.set[:t.begin] {
				c <- p
			}
			for _, p := range t.set[t.end:] {
				c <- p
			}
		}
		close(c)
	}()
	return c
}

func CrossValidate(set []DataSequence, k int) []ValidationSet {
	l := len(set)
	out := make([]ValidationSet, k)
	for i := 0; i < k; i++ {
		begin, end := (i*l) / k, ((i+1)*l) / k
		out[i] = ValidationSet{
			set: set,
			begin: begin,
			end: end,
		}
	}
	return out
}

func SprintSeq(pred Predictor, seq DataSequence) string {
	out := ""
	for _, x := range seq.Xs {
		out += fmt.Sprintf("pred(%v) = %f\n", x, pred.F(x))
	}
	out += fmt.Sprintf("Outcome: %f", seq.Z)
	return out
}

func PredictionError(pred Predictor, seq DataSequence) float64 {
	var ss float64
	for _, x := range seq.Xs {
		ss += math.Pow(seq.Z-pred.F(x), 2)
	}
	return ss / 2
}

// Calculate prediction errors (MSE)
func PredictionErrors(pred Predictor, data DataStream, n int) (mse float64) {
	var sumsq float64
	i := 0

	for seq := range data {
		sumsq += PredictionError(pred, seq) / float64(len(seq.Xs))

		i += 1
		if i >= n {
			break
		}
	}

	return sumsq / float64(i)
}