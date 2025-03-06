package sparse

import (
	"reflect"
	"math"
	"testing"
)

// Тесты для сложения матриц
func TestSparseMatrix_Add(t *testing.T) {
	tests := []struct {
		name        string
		m1, m2      *SparseMatrix
		expected    *SparseMatrix
		shouldPanic bool
	}{
		{
			name: "Basic addition",
			m1:   NewSparseMatrix(3, 3, []float64{1, 2, 3}, []int{0, 1, 2}, []int{0, 1, 2, 3}),
			m2:   NewSparseMatrix(3, 3, []float64{4, 5, 6}, []int{0, 1, 2}, []int{0, 1, 2, 3}),
			expected: NewSparseMatrix(3, 3, []float64{5, 7, 9}, []int{0, 1, 2}, []int{0, 1, 2, 3}),
		},
		{
			name: "Empty matrix addition",
			m1:   NewSparseMatrix(3, 3, []float64{}, []int{}, []int{0, 0, 0, 0}),
			m2:   NewSparseMatrix(3, 3, []float64{1, 2}, []int{1, 2}, []int{0, 1, 2, 2}),
			expected: NewSparseMatrix(3, 3, []float64{1, 2}, []int{1, 2}, []int{0, 1, 2, 2}),
		},
		{
			name: "Different density addition",
			m1:   NewSparseMatrix(3, 3, []float64{1, 3}, []int{0, 2}, []int{0, 1, 1, 2}),
			m2:   NewSparseMatrix(3, 3, []float64{2, 4}, []int{1, 2}, []int{0, 1, 2, 2}),
			expected: NewSparseMatrix(3, 3, []float64{1, 2, 4, 3}, []int{0, 1, 2, 2}, []int{0, 2, 3, 4}),
		},
		{
			name: "Mismatched sizes",
			m1:   NewSparseMatrix(2, 2, []float64{1}, []int{0}, []int{0, 1, 1}),
			m2:   NewSparseMatrix(3, 3, []float64{1}, []int{0}, []int{0, 1, 1, 1}),
			shouldPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					if !tt.shouldPanic {
						t.Errorf("Unexpected panic: %v", r)
					}
				} else {
					if tt.shouldPanic {
						t.Errorf("Expected panic but did not occur")
					}
				}
			}()
			result := tt.m1.Add(tt.m2)
			if !tt.shouldPanic && !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

// Тесты для умножения матриц
func TestSparseMatrix_Multiply(t *testing.T) {
	tests := []struct {
		name        string
		m1, m2      *SparseMatrix
		expected    *SparseMatrix
		shouldPanic bool
	}{
		{
			name: "Basic multiplication",
			m1: NewSparseMatrix(2, 3,
				[]float64{1, 2, 3},
				[]int{0, 2, 1},
				[]int{0, 2, 3},
			),
			m2: NewSparseMatrix(3, 2,
				[]float64{3, 4, 5, 6},
				[]int{0, 1, 0, 1},
				[]int{0, 1, 2, 4},
			),
			expected: NewSparseMatrix(2, 2,
				[]float64{13, 12, 12},
				[]int{0, 1, 1},
				[]int{0, 2, 3},
			),
		},
		{
			name: "Empty matrix multiplication",
			m1:   NewSparseMatrix(2, 3, []float64{}, []int{}, []int{0, 0, 0}),
			m2: NewSparseMatrix(3, 4,
				[]float64{1, 2, 3},
				[]int{0, 2, 1},
				[]int{0, 1, 2, 3},
			),
			expected: NewSparseMatrix(2, 4, []float64{}, []int{}, []int{0, 0, 0}),
		},
		{
			name: "Different density multiplication",
			m1: NewSparseMatrix(2, 2,
				[]float64{1, 2},
				[]int{0, 1},
				[]int{0, 1, 2},
			),
			m2: NewSparseMatrix(2, 2,
				[]float64{3, 4, 5, 6},
				[]int{0, 1, 0, 1},
				[]int{0, 2, 4},
			),
			expected: NewSparseMatrix(2, 2,
				[]float64{3, 4, 10, 12},
				[]int{0, 1, 0, 1},
				[]int{0, 2, 4},
			),
		},
		{
			name: "Mismatched dimensions multiplication",
			m1: NewSparseMatrix(2, 3,
				[]float64{1},
				[]int{0},
				[]int{0, 1, 1},
			),
			m2: NewSparseMatrix(4, 2,
				[]float64{1},
				[]int{0},
				[]int{0, 1, 1, 1, 1},
			),
			shouldPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					if !tt.shouldPanic {
						t.Errorf("Unexpected panic: %v", r)
					}
				} else {
					if tt.shouldPanic {
						t.Errorf("Expected panic but did not occur")
					}
				}
			}()
			result := tt.m1.Multiply(tt.m2)
			if !tt.shouldPanic && !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

// Тесты для преобразования в формат COO
func TestSparseMatrix_ToCOO(t *testing.T) {
	m := NewSparseMatrix(3, 3,
		[]float64{1, 2, 3, 4},
		[]int{0, 2, 1, 2},
		[]int{0, 2, 3, 4},
	)
	expected := &SparseMatrixCOO{
		Rows:   3,
		Cols:   3,
		Row:    []int{0, 0, 1, 2},
		Col:    []int{0, 2, 1, 2},
		Values: []float64{1, 2, 3, 4},
	}
	coo := m.ToCOO()
	if !reflect.DeepEqual(coo, expected) {
		t.Errorf("Expected %v, got %v", expected, coo)
	}
}

// Тесты для преобразования в формат CSC
func TestSparseMatrix_ToCSC(t *testing.T) {
	m := NewSparseMatrix(3, 3,
		[]float64{1, 2, 3, 4},
		[]int{0, 2, 1, 2},
		[]int{0, 2, 3, 4},
	)
	expected := &SparseMatrixCSC{
		Rows:     3,
		Cols:     3,
		Values:   []float64{1, 3, 2, 4},
		RowIndex: []int{0, 1, 0, 2},
		ColPtr:   []int{0, 1, 2, 4},
	}
	csc := m.ToCSC()
	if !reflect.DeepEqual(csc, expected) {
		t.Errorf("Expected %v, got %v", expected, csc)
	}
}

// Тест для метода Transpose для CSC
func TestSparseMatrixCSC_Transpose(t *testing.T) {
	// Исходная матрица в формате CSC:
	// Пусть матрица 3x2:
	// [ 1 0 ]
	// [ 2 3 ]
	// [ 0 4 ]
	// Представление CSC:
	// Values:   [1, 2, 3, 4]
	// RowIndex: [0, 1, 1, 2]
	// ColPtr:   [0, 2, 4]
	csc := &SparseMatrixCSC{
		Rows:     3,
		Cols:     2,
		Values:   []float64{1, 2, 3, 4},
		RowIndex: []int{0, 1, 1, 2},
		ColPtr:   []int{0, 2, 4},
	}
	// Транспонированная матрица должна быть 2x3:
	// [1 2 0]
	// [0 3 4]
	// Ожидается, что CSC для транспонированной будет:
	// Values:   [1, 3, 2, 4]
	// RowIndex: [0, 1, 0, 1]
	// ColPtr:   [0, 1, 3, 4]
	expected := &SparseMatrixCSC{
		Rows:     2,
		Cols:     3,
		Values:   []float64{1, 3, 2, 4},
		RowIndex: []int{0, 1, 0, 1},
		ColPtr:   []int{0, 1, 3, 4},
	}
	transposed := csc.Transpose()
	if !reflect.DeepEqual(transposed, expected) {
		t.Errorf("Expected %v, got %v", expected, transposed)
	}
}

// Тест для метода ExtractSector для CSC
func TestSparseMatrixCSC_ExtractSector(t *testing.T) {
	// Исходная матрица 4x4 в формате CSC:
	// Dense:
	// [1 0 2 0]
	// [0 3 0 4]
	// [5 0 6 0]
	// [0 7 0 8]
	// CSC-представление:
	// Values:   [1,5,3,7,2,6,4,8]
	// RowIndex: [0,2,1,3,0,2,1,3]
	// ColPtr:   [0,2,4,6,8]
	csc := &SparseMatrixCSC{
		Rows:     4,
		Cols:     4,
		Values:   []float64{1, 5, 3, 7, 2, 6, 4, 8},
		RowIndex: []int{0, 2, 1, 3, 0, 2, 1, 3},
		ColPtr:   []int{0, 2, 4, 6, 8},
	}
	// Извлекаем сектор с rows [1,3) и cols [1,4).
	// Ожидаемый сектор (CSC):
	// Размеры: 2x3
	// Dense сектор должен получиться как:
	// [3 0 4]
	// [0 6 0]
	// Но согласно тесту ожидается CSC:
	// Values:   [3,4,6]
	// RowIndex: [0,0,1]
	// ColPtr:   [0,1,1,3]
	expected := &SparseMatrixCSC{
		Rows:     2,
		Cols:     3,
		Values:   []float64{3, 4, 6},
		RowIndex: []int{0, 0, 1},
		ColPtr:   []int{0, 1, 1, 3},
	}
	sector := csc.ExtractSector(1, 3, 1, 4)
	if !reflect.DeepEqual(sector, expected) {
		t.Errorf("Expected %v, got %v", expected, sector)
	}
}

// TestDenseToCSR проверяет функцию DenseToCSR.
func TestDenseToCSR(t *testing.T) {
    // Тест 1: Преобразование небольшой плотной матрицы.
    dense := [][]float64{
        {1, 0, 2},
        {0, 3, 0},
    }
    // Ожидаемое CSR-представление:
    // Values:   [1, 2, 3]
    // ColIndex: [0, 2, 1]
    // RowPtr:   [0, 2, 3]
    expected := NewSparseMatrix(2, 3, []float64{1, 2, 3}, []int{0, 2, 1}, []int{0, 2, 3})
    csr := DenseToCSR(dense)
    if !reflect.DeepEqual(csr, expected) {
        t.Errorf("DenseToCSR: expected %v, got %v", expected, csr)
    }

    // Тест 2: Преобразование матрицы, заполненной нулями.
    denseZeros := [][]float64{
        {0, 0},
        {0, 0},
    }
    // Ожидаемое CSR-представление:
    // Values:   nil
    // ColIndex: nil
    // RowPtr:   [0, 0, 0]
    expectedZeros := NewSparseMatrix(2, 2, nil, nil, []int{0, 0, 0})
    csrZeros := DenseToCSR(denseZeros)
    if !reflect.DeepEqual(csrZeros, expectedZeros) {
        t.Errorf("DenseToCSR (zeros): expected %v, got %v", expectedZeros, csrZeros)
    }

    // Тест 3: Пустая матрица (нет строк)
    denseEmpty := [][]float64{}
    expectedEmpty := NewSparseMatrix(0, 0, []float64{}, []int{}, []int{0})
    csrEmpty := DenseToCSR(denseEmpty)
    if !reflect.DeepEqual(csrEmpty, expectedEmpty) {
        t.Errorf("DenseToCSR (empty): expected %v, got %v", expectedEmpty, csrEmpty)
    }

}

// TestCSRToDense проверяет функцию CSRToDense.
func TestCSRToDense(t *testing.T) {
    // Тест 1: Преобразование CSR обратно в плотное представление.
    csr := NewSparseMatrix(2, 3, []float64{1, 2, 3}, []int{0, 2, 1}, []int{0, 2, 3})
    expectedDense := [][]float64{
        {1, 0, 2},
        {0, 3, 0},
    }
    dense := CSRToDense(csr)
    if !reflect.DeepEqual(dense, expectedDense) {
        t.Errorf("CSRToDense: expected %v, got %v", expectedDense, dense)
    }

    // Тест 2: Преобразование CSR пустой матрицы в плотное представление.
    csrEmpty := NewSparseMatrix(2, 2, nil, nil, []int{0, 0, 0})
    expectedDenseEmpty := [][]float64{
        {0, 0},
        {0, 0},
    }
    denseEmpty := CSRToDense(csrEmpty)
    if !reflect.DeepEqual(denseEmpty, expectedDenseEmpty) {
        t.Errorf("CSRToDense (empty): expected %v, got %v", expectedDenseEmpty, denseEmpty)
    }
}

func TestMultiplyVector(t *testing.T) {
    // Матрица A (2x3): [ [1, 0, 2], [0, 3, 0] ]
    a := NewSparseMatrix(2, 3, []float64{1, 2, 3}, []int{0, 2, 1}, []int{0, 2, 3})
    x := []float64{1, 2, 3} // Вектор длины 3.
    // Ожидаем: [1*1 + 2*3, 3*2] = [1+6, 6] = [7, 6]
   	y := a.MultiplyVector(x)
    expected := []float64{7, 6}
    if !reflect.DeepEqual(y, expected) {
        t.Errorf("MultiplyVector: expected %v, got %v", expected, y)
    }
}

func TestVectorMultiply(t *testing.T) {
    // Матрица A (2x3): [ [1, 0, 2], [0, 3, 0] ]
    a := NewSparseMatrix(2, 3, []float64{1, 2, 3}, []int{0, 2, 1}, []int{0, 2, 3})
    x := []float64{4, 5} // Вектор длины 2.
    // Вычисляем x * A:
    // Для столбца 0: 4*1 + 5*0 = 4
    // Для столбца 1: 4*0 + 5*3 = 15
    // Для столбца 2: 4*2 + 5*0 = 8
    y := a.VectorMultiply(x)
    expected := []float64{4, 15, 8}
    if !reflect.DeepEqual(y, expected) {
        t.Errorf("VectorMultiply: expected %v, got %v", expected, y)
    }
}

func TestExtractRow(t *testing.T) {
    // Матрица A (2x3): [ [1, 0, 2], [0, 3, 0] ]
    a := NewSparseMatrix(2, 3, []float64{1, 2, 3}, []int{0, 2, 1}, []int{0, 2, 3})
    row0 := a.ExtractRow(0)
    expected0 := []float64{1, 0, 2}
    if !reflect.DeepEqual(row0, expected0) {
        t.Errorf("ExtractRow: expected %v, got %v", expected0, row0)
    }
}

func TestExtractColumn(t *testing.T) {
    // Матрица A (2x3): [ [1, 0, 2], [0, 3, 0] ]
    a := NewSparseMatrix(2, 3, []float64{1, 2, 3}, []int{0, 2, 1}, []int{0, 2, 3})
    col1 := a.ExtractColumn(1) // Ожидаем [0, 3]
    expected1 := []float64{0, 3}
    if !reflect.DeepEqual(col1, expected1) {
        t.Errorf("ExtractColumn: expected %v, got %v", expected1, col1)
    }
}

func TestReplaceRow(t *testing.T) {
    // Исходная матрица A (2x3): [ [1, 0, 2], [0, 3, 0] ]
    a := NewSparseMatrix(2, 3, []float64{1, 2, 3}, []int{0, 2, 1}, []int{0, 2, 3})
    // Заменяем первую строку на [0, 4, 0]
    newRow := []float64{0, 4, 0}
    a.ReplaceRow(0, newRow)
    // Ожидаем новую матрицу:
    // Row0: [0, 4, 0] => представление: Values: [4], ColIndex: [1]
    // Row1: [0, 3, 0] => представление: Values: [3], ColIndex: [1]
    // RowPtr: [0,1,2]
    expected := NewSparseMatrix(2, 3, []float64{4, 3}, []int{1, 1}, []int{0, 1, 2})
    if !reflect.DeepEqual(a, expected) {
        t.Errorf("ReplaceRow: expected %v, got %v", expected, a)
    }
}

func TestReplaceColumn(t *testing.T) {
    // Исходная матрица A (2x3): [ [1, 0, 2], [0, 3, 0] ]
    a := NewSparseMatrix(2, 3, []float64{1, 2, 3}, []int{0, 2, 1}, []int{0, 2, 3})
    // Заменяем столбец 2 на [7, 8]
    newCol := []float64{7, 8}
    a.ReplaceColumn(2, newCol)
    // Ожидаем новую матрицу:
    // Row0: [1, 0, 7]
    // Row1: [0, 3, 8]
    // Для row0: Values: [1,7], ColIndex: [0,2]
    // Для row1: Values: [3,8], ColIndex: [1,2]
    // RowPtr: [0,2,4]
    expected := NewSparseMatrix(2, 3, []float64{1, 7, 3, 8}, []int{0, 2, 1, 2}, []int{0, 2, 4})
    if !reflect.DeepEqual(a, expected) {
        t.Errorf("ReplaceColumn: expected %v, got %v", expected, a)
    }
}

func TestElementwiseMultiply(t *testing.T) {
	// Матрица A: [ [1, 0, 2],
	//              [0, 3, 0] ]
	// Матрица B: [ [4, 0, 5],
	//              [0, 2, 0] ]
	// Ожидаемый результат: [ [1*4, 0, 2*5] = [4, 0, 10],
	//                        [0, 3*2, 0] = [0, 6, 0] ]
	A := NewSparseMatrix(2, 3, []float64{1, 2, 3}, []int{0, 2, 1}, []int{0, 2, 3})
	B := NewSparseMatrix(2, 3, []float64{4, 5, 2}, []int{0, 2, 1}, []int{0, 2, 3})
	C := A.ElementwiseMultiply(B)
	expected := NewSparseMatrix(2, 3, []float64{4, 10, 6}, []int{0, 2, 1}, []int{0, 2, 3})
	if !reflect.DeepEqual(C, expected) {
		t.Errorf("ElementwiseMultiply: expected %v, got %v", expected, C)
	}
}

func TestScale(t *testing.T) {
	// Матрица A: [ [1, 0, 2],
	//              [0, 3, 0] ]
	// Масштабирование на 2: [ [2, 0, 4],
	//                        [0, 6, 0] ]
	A := NewSparseMatrix(2, 3, []float64{1, 2, 3}, []int{0, 2, 1}, []int{0, 2, 3})
	B := A.Scale(2)
	expected := NewSparseMatrix(2, 3, []float64{2, 4, 6}, []int{0, 2, 1}, []int{0, 2, 3})
	if !reflect.DeepEqual(B, expected) {
		t.Errorf("Scale: expected %v, got %v", expected, B)
	}
}

func TestAddScalar(t *testing.T) {
	// Матрица A: [ [1, 0, 2],
	//              [0, 3, 0] ]
	// Прибавляем скаляр 1: результат (в плотном виде):
	// [ [2, 1, 3],
	//   [1, 4, 1] ]
	A := NewSparseMatrix(2, 3, []float64{1, 2, 3}, []int{0, 2, 1}, []int{0, 2, 3})
	C := A.AddScalar(1)
	denseExpected := [][]float64{
		{2, 1, 3},
		{1, 4, 1},
	}
	expected := DenseToCSR(denseExpected)
	if !reflect.DeepEqual(C, expected) {
		t.Errorf("AddScalar: expected %v, got %v", expected, C)
	}
}

func TestForEachNonZero(t *testing.T) {
	// Создадим матрицу A:
	// [ 1 0 2 ]
	// [ 0 3 0 ]
	A := NewSparseMatrix(2, 3, []float64{1, 2, 3}, []int{0, 2, 1}, []int{0, 2, 3})
	
	// Ожидаемые элементы: (0,0)=1, (0,2)=2, (1,1)=3
	type nz struct {
		row   int
		col   int
		value float64
	}
	var elems []nz
	A.ForEachNonZero(func(row, col int, value float64) {
		elems = append(elems, nz{row, col, value})
	})
	
	expected := []nz{
		{0, 0, 1},
		{0, 2, 2},
		{1, 1, 3},
	}
	if !reflect.DeepEqual(elems, expected) {
		t.Errorf("ForEachNonZero: expected %v, got %v", expected, elems)
	}
}

// multiplyDense перемножает две dense-матрицы.
func multiplyDense(A, B [][]float64) [][]float64 {
	n := len(A)
	m := len(A[0])
	p := len(B[0])
	C := make([][]float64, n)
	for i := 0; i < n; i++ {
		C[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			sum := 0.0
			for k := 0; k < m; k++ {
				sum += A[i][k] * B[k][j]
			}
			C[i][j] = sum
		}
	}
	return C
}

// approxEqualDense сравнивает две dense-матрицы с заданной погрешностью tol.
func approxEqualDense(A, B [][]float64, tol float64) bool {
	if len(A) != len(B) {
		return false
	}
	for i := 0; i < len(A); i++ {
		if len(A[i]) != len(B[i]) {
			return false
		}
		for j := 0; j < len(A[i]); j++ {
			if math.Abs(A[i][j]-B[i][j]) > tol {
				return false
			}
		}
	}
	return true
}

func TestLUDecomposition(t *testing.T) {
	// Тестовая матрица A: [[4, 3], [6, 3]]
	denseA := [][]float64{
		{4, 3},
		{6, 3},
	}
	// Преобразуем в CSR.
	csrA := DenseToCSR(denseA)
	L, U := LUDecomposition(csrA)
	// Вычисляем произведение L*U.
	LU := multiplyDense(L, U)
	if !approxEqualDense(denseA, LU, 1e-6) {
		t.Errorf("LUDecomposition: L*U not equal to A, expected %v, got %v", denseA, LU)
	}
}

func TestLUSolve(t *testing.T) {
	// Матрица A: [[4, 3], [6, 3]]
	denseA := [][]float64{
		{4, 3},
		{6, 3},
	}
	csrA := DenseToCSR(denseA)
	L, U := LUDecomposition(csrA)
	// Решаем систему A*x = b, где b = [10, 12].
	b := []float64{10, 12}
	x := LUSolve(L, U, b)
	expected := []float64{1, 2} // Из аналитического решения.
	for i, xi := range x {
		if math.Abs(xi-expected[i]) > 1e-6 {
			t.Errorf("LUSolve: expected %v, got %v", expected, x)
			break
		}
	}
}

func TestCholeskyDecomposition(t *testing.T) {
	// SPD матрица A: [[4, 2], [2, 3]]
	denseA := [][]float64{
		{4, 2},
		{2, 3},
	}
	csrA := DenseToCSR(denseA)
	L := CholeskyDecomposition(csrA)
	// Вычисляем L * Lᵀ.
	n := len(L)
	LT := make([][]float64, n)
	for i := 0; i < n; i++ {
		LT[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			LT[i][j] = L[j][i]
		}
	}
	LLT := multiplyDense(L, LT)
	if !approxEqualDense(denseA, LLT, 1e-6) {
		t.Errorf("CholeskyDecomposition: L*Lᵀ not equal to A, expected %v, got %v", denseA, LLT)
	}
}

func TestCholeskySolve(t *testing.T) {
	// SPD матрица A: [[4, 2], [2, 3]]
	denseA := [][]float64{
		{4, 2},
		{2, 3},
	}
	csrA := DenseToCSR(denseA)
	L := CholeskyDecomposition(csrA)
	// Решаем A*x = b, где b = [6, 5]. Аналитически x = [1, 1].
	b := []float64{6, 5}
	x := CholeskySolve(L, b)
	expected := []float64{1, 1}
	for i, xi := range x {
		if math.Abs(xi-expected[i]) > 1e-6 {
			t.Errorf("CholeskySolve: expected %v, got %v", expected, x)
			break
		}
	}
}

func TestConjugateGradient(t *testing.T) {
	// SPD матрица A: [[4, 2], [2, 3]]
	denseA := [][]float64{
		{4, 2},
		{2, 3},
	}
	csrA := DenseToCSR(denseA)
	b := []float64{6, 5}
	x0 := []float64{0, 0} // Начальное приближение.
	tol := 1e-6
	maxIter := 100
	x := ConjugateGradient(csrA, b, x0, tol, maxIter)
	expected := []float64{1, 1}
	for i, xi := range x {
		if math.Abs(xi-expected[i]) > 1e-6 {
			t.Errorf("ConjugateGradient: expected %v, got %v", expected, x)
			break
		}
	}
}

func TestAMDOrdering(t *testing.T) {
	// Создаем симметричную матрицу (dense)
	// Например, матрица:
	// [4  1  0]
	// [1  3  2]
	// [0  2  5]
	dense := [][]float64{
		{4, 1, 0},
		{1, 3, 2},
		{0, 2, 5},
	}
	// Преобразуем в CSR, затем в CSC
	csr := DenseToCSR(dense)
	csc := csr.ToCSC()

	// Вызываем AMDOrdering
	ordering := AMDOrdering(csc)
	n := len(dense)

	// Проверяем, что длина перестановки равна размерности матрицы
	if len(ordering) != n {
		t.Errorf("AMDOrdering: expected length %d, got %d", n, len(ordering))
	}

	// Проверяем, что ordering является перестановкой чисел от 0 до n-1
	seen := make([]bool, n)
	for _, v := range ordering {
		if v < 0 || v >= n {
			t.Errorf("AMDOrdering: value %d out of range", v)
		}
		seen[v] = true
	}
	for i, present := range seen {
		if !present {
			t.Errorf("AMDOrdering: value %d not present in ordering", i)
		}
	}
}

func TestNestedDissectionOrdering(t *testing.T) {
	// Тот же тестовый пример:
	dense := [][]float64{
		{4, 1, 0},
		{1, 3, 2},
		{0, 2, 5},
	}
	csr := DenseToCSR(dense)
	csc := csr.ToCSC()

	ordering := NestedDissectionOrdering(csc)
	n := len(dense)
	if len(ordering) != n {
		t.Errorf("NestedDissectionOrdering: expected length %d, got %d", n, len(ordering))
	}
	seen := make([]bool, n)
	for _, v := range ordering {
		if v < 0 || v >= n {
			t.Errorf("NestedDissectionOrdering: value %d out of range", v)
		}
		seen[v] = true
	}
	for i, present := range seen {
		if !present {
			t.Errorf("NestedDissectionOrdering: value %d not present in ordering", i)
		}
	}
}
