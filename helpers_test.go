package nnalgs

import (
	"testing"
	"reflect"
)

func list(d DataStream) (out []DataSequence) {
	for x := range d {
		out = append(out, x)
	}
	return
}

func TestCrossValidate(t *testing.T) {
	full := []DataSequence{
		DataSequence{[][]float64{{1}}, 1},
		DataSequence{[][]float64{{2}}, 2},
		DataSequence{[][]float64{{3}}, 3},
		DataSequence{[][]float64{{4}}, 4},
		DataSequence{[][]float64{{5}}, 5},
	}

	split := CrossValidate(full, 3)

	t.Log(split)

	if len(list(split[0].Test())) != 1 {
		t.Error("First split test has wrong size")
	}
	if !reflect.DeepEqual(append(list(split[0].Test()), list(split[0].Train(1))...), full) {
		t.Log(append(list(split[0].Test()), list(split[0].Train(1))...))
		t.Error("First split combined is not full")
	}
	if len(list(split[1].Train(2))) != 6 {
		t.Error("Second split train x2 has wrong size")
	}
	if len(list(split[2].Test())) != 2 {
		t.Error("Third split test has wrong size")
	}
}
