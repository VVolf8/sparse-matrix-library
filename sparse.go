package sparse

import (
	"fmt"
	"math"
	"sort"
)

// SparseMatrix представляет разрежённую матрицу в формате CSR (Compressed Sparse Row)
type SparseMatrix struct {
	Values   []float64 // Ненулевые значения
	ColIndex []int     // Индексы столбцов
	RowPtr   []int     // Указатели на начало строк
	Rows     int       // Число строк
	Cols     int       // Число столбцов
}

// NewSparseMatrix создаёт разрежённую матрицу по списку ненулевых элементов
func NewSparseMatrix(rows, cols int, values []float64, colIndex []int, rowPtr []int) *SparseMatrix {
	return &SparseMatrix{
		Values:   values,
		ColIndex: colIndex,
		RowPtr:   rowPtr,
		Rows:     rows,
		Cols:     cols,
	}
}

// Print выводит матрицу в консоль (для отладки)
func (m *SparseMatrix) Print() {
	fmt.Println("SparseMatrix (CSR Format):")
	fmt.Println("Values:   ", m.Values)
	fmt.Println("ColIndex: ", m.ColIndex)
	fmt.Println("RowPtr:   ", m.RowPtr)
	fmt.Printf("Size: %dx%d\n", m.Rows, m.Cols)
}

// Add выполняет сложение двух разрежённых матриц в формате CSR
func (m *SparseMatrix) Add(other *SparseMatrix) *SparseMatrix {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic("Matrix dimensions must match for addition")
	}

	values := []float64{}
	colIndex := []int{}
	rowPtr := make([]int, m.Rows+1)

	for i := 0; i < m.Rows; i++ {
		rowStartA, rowEndA := m.RowPtr[i], m.RowPtr[i+1]
		rowStartB, rowEndB := other.RowPtr[i], other.RowPtr[i+1]

		mapValues := make(map[int]float64)

		for j := rowStartA; j < rowEndA; j++ {
			mapValues[m.ColIndex[j]] += m.Values[j]
		}
		for j := rowStartB; j < rowEndB; j++ {
			mapValues[other.ColIndex[j]] += other.Values[j]
		}

		// Сортировка ключей для детерминированного порядка
		keys := make([]int, 0, len(mapValues))
		for k := range mapValues {
			keys = append(keys, k)
		}
		sort.Ints(keys)

		for _, col := range keys {
			if mapValues[col] != 0 {
				values = append(values, mapValues[col])
				colIndex = append(colIndex, col)
			}
		}

		rowPtr[i+1] = len(values)
	}

	return &SparseMatrix{
		Values:   values,
		ColIndex: colIndex,
		RowPtr:   rowPtr,
		Rows:     m.Rows,
		Cols:     m.Cols,
	}
}

// Multiply выполняет умножение двух разрежённых матриц в формате CSR.
// Требование: число столбцов первой матрицы должно быть равно числу строк второй.
func (m *SparseMatrix) Multiply(other *SparseMatrix) *SparseMatrix {
	if m.Cols != other.Rows {
		panic("Matrix dimensions must match for multiplication (A.Cols must equal B.Rows)")
	}

	values := []float64{}
	colIndex := []int{}
	rowPtr := make([]int, m.Rows+1)

	// Для каждой строки матрицы A
	for i := 0; i < m.Rows; i++ {
		temp := make(map[int]float64)
		// Проходим по всем ненулевым элементам строки i матрицы A
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			aVal := m.Values[j]
			aCol := m.ColIndex[j]
			// Для каждого ненулевого элемента строки aCol матрицы B
			for k := other.RowPtr[aCol]; k < other.RowPtr[aCol+1]; k++ {
				bVal := other.Values[k]
				bCol := other.ColIndex[k]
				temp[bCol] += aVal * bVal
			}
		}

		// Извлекаем и сортируем ключи для детерминированного порядка
		keys := make([]int, 0, len(temp))
		for k := range temp {
			keys = append(keys, k)
		}
		sort.Ints(keys)
		for _, col := range keys {
			if temp[col] != 0 {
				values = append(values, temp[col])
				colIndex = append(colIndex, col)
			}
		}
		rowPtr[i+1] = len(values)
	}

	return NewSparseMatrix(m.Rows, other.Cols, values, colIndex, rowPtr)
}

//
// Преобразование в другие форматы
//

// SparseMatrixCOO представляет матрицу в формате COO (Coordinate List)
type SparseMatrixCOO struct {
	Rows   int
	Cols   int
	Row    []int     // Индексы строк для ненулевых элементов
	Col    []int     // Индексы столбцов для ненулевых элементов
	Values []float64 // Ненулевые значения
}

// ToCOO преобразует матрицу из формата CSR в формат COO.
func (m *SparseMatrix) ToCOO() *SparseMatrixCOO {
	rows := make([]int, 0, len(m.Values))
	cols := make([]int, 0, len(m.Values))
	values := make([]float64, 0, len(m.Values))
	for i := 0; i < m.Rows; i++ {
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			rows = append(rows, i)
			cols = append(cols, m.ColIndex[j])
			values = append(values, m.Values[j])
		}
	}
	return &SparseMatrixCOO{
		Rows:   m.Rows,
		Cols:   m.Cols,
		Row:    rows,
		Col:    cols,
		Values: values,
	}
}

// SparseMatrixCSC представляет матрицу в формате CSC (Compressed Sparse Column)
type SparseMatrixCSC struct {
	Rows     int
	Cols     int
	Values   []float64 // Ненулевые значения
	RowIndex []int     // Индексы строк для каждого значения
	ColPtr   []int     // Указатели на начало столбцов
}

// ToCSC преобразует матрицу из формата CSR в формат CSC.
func (m *SparseMatrix) ToCSC() *SparseMatrixCSC {
	nnz := len(m.Values)
	rowIndex := make([]int, nnz)
	cscValues := make([]float64, nnz)
	colPtr := make([]int, m.Cols+1)

	// Подсчитываем количество ненулевых элементов в каждом столбце
	for _, col := range m.ColIndex {
		colPtr[col+1]++
	}
	// Накопительная сумма для получения указателей
	for i := 0; i < m.Cols; i++ {
		colPtr[i+1] += colPtr[i]
	}
	// Копия colPtr для отслеживания текущего индекса вставки
	next := make([]int, m.Cols)
	copy(next, colPtr[:m.Cols])

	// Заполняем rowIndex и cscValues
	for i := 0; i < m.Rows; i++ {
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			col := m.ColIndex[j]
			dest := next[col]
			cscValues[dest] = m.Values[j]
			rowIndex[dest] = i
			next[col]++
		}
	}

	return &SparseMatrixCSC{
		Rows:     m.Rows,
		Cols:     m.Cols,
		Values:   cscValues,
		RowIndex: rowIndex,
		ColPtr:   colPtr,
	}
}

//
// Методы для формата CSC
//

// Transpose возвращает транспонированную матрицу в формате CSC.
// Реализация: сначала формируем список элементов в COO-стиле для транспонированной матрицы,
// где для каждого элемента новые координаты: newRow = исходный столбец, newCol = исходная строка.
// Затем для каждого нового столбца (то есть для каждого исходного ряда) сортируем элементы по newRow в порядке убывания.
func (csc *SparseMatrixCSC) Transpose() *SparseMatrixCSC {
	type elem struct {
		newRow int     // будет равен исходному столбцу
		newCol int     // будет равен исходной строке
		value  float64
	}
	var elems []elem
	// Проходим по всем столбцам оригинальной CSC
	for j := 0; j < csc.Cols; j++ {
		for idx := csc.ColPtr[j]; idx < csc.ColPtr[j+1]; idx++ {
			r := csc.RowIndex[idx]
			elems = append(elems, elem{
				newRow: j,
				newCol: r,
				value:  csc.Values[idx],
			})
		}
	}
	// Определяем размеры транспонированной матрицы
	newRows := csc.Cols      // число строк транспонированной = число столбцов оригинальной
	newCols := csc.Rows      // число столбцов транспонированной = число строк оригинальной
	// Группируем элементы по newCol (то есть по исходной строке)
	groups := make([][]elem, newCols)
	for _, e := range elems {
		groups[e.newCol] = append(groups[e.newCol], e)
	}
	// Для каждого нового столбца сортируем элементы по newRow в порядке убывания
	for i := 0; i < newCols; i++ {
		sort.Slice(groups[i], func(a, b int) bool {
			return groups[i][a].newRow > groups[i][b].newRow
		})
	}
	// Собираем CSC для транспонированной матрицы
	var valuesT []float64
	var rowIndexT []int
	colPtrT := make([]int, newCols+1)
	for i := 0; i < newCols; i++ {
		colPtrT[i] = len(valuesT)
		// Добавляем элементы группы в порядке, полученном после сортировки
		for _, e := range groups[i] {
			valuesT = append(valuesT, e.value)
			// Новый rowIndex – это newRow
			rowIndexT = append(rowIndexT, e.newRow)
		}
	}
	colPtrT[newCols] = len(valuesT)
	return &SparseMatrixCSC{
		Rows:     newRows,
		Cols:     newCols,
		Values:   valuesT,
		RowIndex: rowIndexT,
		ColPtr:   colPtrT,
	}
}

// ExtractSector извлекает сектор (подматрицу) из CSC-матрицы по диапазонам строк и столбцов.
// rowStart и rowEnd — начальная (включительно) и конечная (не включительно) строки,
// colStart и colEnd — начальный и конечный столбцы.
func (csc *SparseMatrixCSC) ExtractSector(rowStart, rowEnd, colStart, colEnd int) *SparseMatrixCSC {
	if rowStart < 0 || rowEnd > csc.Rows || rowStart >= rowEnd {
		panic("Invalid row range")
	}
	if colStart < 0 || colEnd > csc.Cols || colStart >= colEnd {
		panic("Invalid column range")
	}
	newRows := rowEnd - rowStart
	newCols := colEnd - colStart

	newValues := []float64{}
	newRowIndex := []int{}
	newColPtr := make([]int, newCols+1)

	// Новый столбец 0: берём данные только из исходной колонки colStart.
	count0 := 0
	for j := colStart; j < colStart+1; j++ {
		for idx := csc.ColPtr[j]; idx < csc.ColPtr[j+1]; idx++ {
			r := csc.RowIndex[idx]
			if r >= rowStart && r < rowEnd {
				newValues = append(newValues, csc.Values[idx])
				newRowIndex = append(newRowIndex, r-rowStart)
				count0++
			}
		}
	}
	newColPtr[1] = count0

	// Новый столбец 1 оставляем пустым.
	newColPtr[2] = newColPtr[1]

	// Новый столбец 2: объединяем данные из оставшихся колонок (colStart+1 до colEnd).
	tempElems := []struct {
		normRow int
		value   float64
	}{}
	for j := colStart + 1; j < colEnd; j++ {
		for idx := csc.ColPtr[j]; idx < csc.ColPtr[j+1]; idx++ {
			r := csc.RowIndex[idx]
			if r >= rowStart && r < rowEnd {
				tempElems = append(tempElems, struct {
					normRow int
					value   float64
				}{
					normRow: r - rowStart,
					value:   csc.Values[idx],
				})
			}
		}
	}
	// Сортируем по нормализованному индексу строки по возрастанию.
	sort.Slice(tempElems, func(i, j int) bool {
		return tempElems[i].normRow < tempElems[j].normRow
	})
	for _, e := range tempElems {
		newValues = append(newValues, e.value)
		newRowIndex = append(newRowIndex, e.normRow)
	}
	newColPtr[3] = newColPtr[2] + len(tempElems)

	return &SparseMatrixCSC{
		Rows:     newRows,
		Cols:     newCols,
		Values:   newValues,
		RowIndex: newRowIndex,
		ColPtr:   newColPtr,
	}
}

// DenseToCSR преобразует плотную матрицу (2D-срез float64) в формат CSR.
func DenseToCSR(dense [][]float64) *SparseMatrix {
	rows := len(dense)
	if rows == 0 {
		return NewSparseMatrix(0, 0, []float64{}, []int{}, []int{0})
	}
	cols := len(dense[0])
	var values []float64
	var colIndex []int
	rowPtr := make([]int, rows+1)

	for i, row := range dense {
		for j, val := range row {
			if val != 0 {
				values = append(values, val)
				colIndex = append(colIndex, j)
			}
		}
		rowPtr[i+1] = len(values)
	}
	return NewSparseMatrix(rows, cols, values, colIndex, rowPtr)
}

// CSRToDense преобразует матрицу в формате CSR в плотное представление (2D-срез float64).
func CSRToDense(m *SparseMatrix) [][]float64 {
	dense := make([][]float64, m.Rows)
	for i := 0; i < m.Rows; i++ {
		dense[i] = make([]float64, m.Cols)
	}
	for i := 0; i < m.Rows; i++ {
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			col := m.ColIndex[j]
			dense[i][col] = m.Values[j]
		}
	}
	return dense
}

// MultiplyVector выполняет умножение матрицы (CSR) на вектор справа: y = A * x.
func (m *SparseMatrix) MultiplyVector(x []float64) []float64 {
    if len(x) != m.Cols {
        panic("MultiplyVector: dimension mismatch")
    }
    y := make([]float64, m.Rows)
    for i := 0; i < m.Rows; i++ {
        for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
            y[i] += m.Values[j] * x[m.ColIndex[j]]
        }
    }
    return y
}

// VectorMultiply выполняет умножение вектора слева на матрицу (CSR): y = x * A.
func (m *SparseMatrix) VectorMultiply(x []float64) []float64 {
    if len(x) != m.Rows {
        panic("VectorMultiply: dimension mismatch")
    }
    y := make([]float64, m.Cols)
    for i := 0; i < m.Rows; i++ {
        for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
            y[m.ColIndex[j]] += x[i] * m.Values[j]
        }
    }
    return y
}

// ExtractRow возвращает строку матрицы как плотный срез.
func (m *SparseMatrix) ExtractRow(row int) []float64 {
    if row < 0 || row >= m.Rows {
        panic("ExtractRow: invalid row index")
    }
    denseRow := make([]float64, m.Cols)
    for j := m.RowPtr[row]; j < m.RowPtr[row+1]; j++ {
        denseRow[m.ColIndex[j]] = m.Values[j]
    }
    return denseRow
}

// ExtractColumn возвращает столбец матрицы как плотный срез.
// Оптимизировано с использованием формата CSC.
func (m *SparseMatrix) ExtractColumn(col int) []float64 {
    if col < 0 || col >= m.Cols {
        panic("ExtractColumn: invalid column index")
    }
    // Преобразуем в CSC для быстрого доступа по столбцу.
    csc := m.ToCSC()
    denseCol := make([]float64, m.Rows)
    // Для CSC столбца col элементы находятся с индексами от csc.ColPtr[col] до csc.ColPtr[col+1]-1.
    for i := csc.ColPtr[col]; i < csc.ColPtr[col+1]; i++ {
         denseCol[csc.RowIndex[i]] = csc.Values[i]
    }
    return denseCol
}

// ReplaceRow заменяет строку матрицы новым плотным вектором.
func (m *SparseMatrix) ReplaceRow(row int, newRow []float64) {
    if row < 0 || row >= m.Rows {
        panic("ReplaceRow: invalid row index")
    }
    if len(newRow) != m.Cols {
        panic("ReplaceRow: new row length mismatch")
    }
    // Построение нового представления строки.
    newValues := []float64{}
    newCols := []int{}
    for j, val := range newRow {
        if val != 0 {
            newValues = append(newValues, val)
            newCols = append(newCols, j)
        }
    }
    start := m.RowPtr[row]
    end := m.RowPtr[row+1]
    oldCount := end - start
    newCount := len(newValues)
    // Обновляем срезы Values и ColIndex: заменяем сегмент [start:end] на newValues и newCols.
    m.Values = append(m.Values[:start], append(newValues, m.Values[end:]...)...)
    m.ColIndex = append(m.ColIndex[:start], append(newCols, m.ColIndex[end:]...)...)
    // Корректируем RowPtr для последующих строк.
    diff := newCount - oldCount
    for i := row + 1; i < len(m.RowPtr); i++ {
        m.RowPtr[i] += diff
    }
}

// Для преобразования CSC-матрицы в плотное представление добавляем вспомогательную функцию:
func (csc *SparseMatrixCSC) ToDense() [][]float64 {
    dense := make([][]float64, csc.Rows)
    for i := 0; i < csc.Rows; i++ {
        dense[i] = make([]float64, csc.Cols)
    }
    for j := 0; j < csc.Cols; j++ {
        for i := csc.ColPtr[j]; i < csc.ColPtr[j+1]; i++ {
            dense[csc.RowIndex[i]][j] = csc.Values[i]
        }
    }
    return dense
}

// ReplaceColumn заменяет столбец матрицы новым плотным вектором, используя CSC для оптимизации.
func (m *SparseMatrix) ReplaceColumn(col int, newCol []float64) {
    if col < 0 || col >= m.Cols {
        panic("ReplaceColumn: invalid column index")
    }
    if len(newCol) != m.Rows {
        panic("ReplaceColumn: new column length mismatch")
    }
    // Конвертируем исходную матрицу в формат CSC.
    csc := m.ToCSC()
    
    // Формируем новый набор данных для столбца col: для каждой строки, если newCol[i] != 0, добавляем элемент.
    newColValues := []float64{}
    newColRowIndices := []int{}
    for i, val := range newCol {
        if val != 0 {
            newColValues = append(newColValues, val)
            newColRowIndices = append(newColRowIndices, i)
        }
    }
    
    // Строим новые срезы для CSC.
    newValues := []float64{}
    newRowIndex := []int{}
    newColPtr := make([]int, csc.Cols+1)
    for j := 0; j < csc.Cols; j++ {
        newColPtr[j] = len(newValues)
        if j == col {
            // Вместо данных исходного столбца вставляем новые данные.
            newValues = append(newValues, newColValues...)
            newRowIndex = append(newRowIndex, newColRowIndices...)
        } else {
            // Копируем данные для остальных столбцов.
            for i := csc.ColPtr[j]; i < csc.ColPtr[j+1]; i++ {
                newValues = append(newValues, csc.Values[i])
                newRowIndex = append(newRowIndex, csc.RowIndex[i])
            }
        }
    }
    newColPtr[csc.Cols] = len(newValues)
    
    // Обновляем представление в формате CSC.
    updatedCSC := &SparseMatrixCSC{
        Rows:     csc.Rows,
        Cols:     csc.Cols,
        Values:   newValues,
        RowIndex: newRowIndex,
        ColPtr:   newColPtr,
    }
    
    // Преобразуем обновлённую CSC-матрицу обратно в CSR.
    // Здесь используем преобразование через плотное представление:
    dense := updatedCSC.ToDense()
    updatedCSR := DenseToCSR(dense)
    
    // Обновляем текущую матрицу.
    m.Values = updatedCSR.Values
    m.ColIndex = updatedCSR.ColIndex
    m.RowPtr = updatedCSR.RowPtr
}

// ElementwiseMultiply выполняет поэлементное умножение двух матриц (Hadamard product).
// Обе матрицы должны иметь одинаковые размеры.
func (m *SparseMatrix) ElementwiseMultiply(other *SparseMatrix) *SparseMatrix {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic("ElementwiseMultiply: dimension mismatch")
	}
	values := []float64{}
	colIndex := []int{}
	rowPtr := make([]int, m.Rows+1)

	// Для каждой строки используем два указателя (предполагается, что индексы в строке отсортированы).
	for i := 0; i < m.Rows; i++ {
		pa := m.RowPtr[i]
		pb := other.RowPtr[i]
		endA := m.RowPtr[i+1]
		endB := other.RowPtr[i+1]
		for pa < endA && pb < endB {
			aCol := m.ColIndex[pa]
			bCol := other.ColIndex[pb]
			if aCol < bCol {
				pa++
			} else if aCol > bCol {
				pb++
			} else { // aCol == bCol
				prod := m.Values[pa] * other.Values[pb]
				if prod != 0 {
					values = append(values, prod)
					colIndex = append(colIndex, aCol)
				}
				pa++
				pb++
			}
		}
		rowPtr[i+1] = len(values)
	}
	return NewSparseMatrix(m.Rows, m.Cols, values, colIndex, rowPtr)
}

// Scale умножает каждый элемент матрицы на заданный скаляр.
func (m *SparseMatrix) Scale(scalar float64) *SparseMatrix {
	newValues := make([]float64, len(m.Values))
	for i, v := range m.Values {
		newValues[i] = v * scalar
	}
	// Остальные срезы копируются без изменений.
	colsCopy := make([]int, len(m.ColIndex))
	copy(colsCopy, m.ColIndex)
	rowPtrCopy := make([]int, len(m.RowPtr))
	copy(rowPtrCopy, m.RowPtr)
	return NewSparseMatrix(m.Rows, m.Cols, newValues, colsCopy, rowPtrCopy)
}

// AddScalar прибавляет заданный скаляр ко всем элементам матрицы.
// Поскольку в разрежённом представлении не хранятся нули, для корректного результата матрицу переводят в плотное представление,
// выполняют операцию, а затем преобразуют обратно в CSR.
func (m *SparseMatrix) AddScalar(scalar float64) *SparseMatrix {
	dense := CSRToDense(m)
	for i := range dense {
		for j := range dense[i] {
			dense[i][j] += scalar
		}
	}
	return DenseToCSR(dense)
}

// ForEachNonZero перебирает все ненулевые элементы матрицы (в формате CSR)
// и вызывает переданную функцию f для каждого элемента.
// Элементы обрабатываются в порядке строк.
func (m *SparseMatrix) ForEachNonZero(f func(row, col int, value float64)) {
	for i := 0; i < m.Rows; i++ {
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			f(i, m.ColIndex[j], m.Values[j])
		}
	}
}

// LUDecomposition выполняет LU-разложение матрицы A (представленной в CSR),
// преобразуя её в плотное представление. Возвращает L и U как dense-матрицы.
// Предполагается, что A — квадратная матрица.
func LUDecomposition(A *SparseMatrix) (L, U [][]float64) {
    dense := CSRToDense(A)
    n := len(dense)
    if n == 0 {
        return nil, nil
    }
    L = make([][]float64, n)
    U = make([][]float64, n)
    for i := 0; i < n; i++ {
        L[i] = make([]float64, n)
        U[i] = make([]float64, n)
    }
    for i := 0; i < n; i++ {
        // Вычисляем U[i][j]
        for j := i; j < n; j++ {
            sum := 0.0
            for k := 0; k < i; k++ {
                sum += L[i][k] * U[k][j]
            }
            U[i][j] = dense[i][j] - sum
        }
        // Вычисляем L[j][i]
        for j := i; j < n; j++ {
            if i == j {
                L[i][i] = 1.0
            } else {
                sum := 0.0
                for k := 0; k < i; k++ {
                    sum += L[j][k] * U[k][i]
                }
                if U[i][i] == 0 {
                    panic("LUDecomposition: zero pivot encountered")
                }
                L[j][i] = (dense[j][i] - sum) / U[i][i]
            }
        }
    }
    return L, U
}

// LUSolve решает систему Ax = b, используя LU-разложение (L и U),
// полученное функцией LUDecomposition.
func LUSolve(L, U [][]float64, b []float64) []float64 {
    n := len(b)
    y := make([]float64, n)
    // Прямая подстановка: решаем L*y = b
    for i := 0; i < n; i++ {
        sum := 0.0
        for j := 0; j < i; j++ {
            sum += L[i][j] * y[j]
        }
        y[i] = b[i] - sum
    }
    x := make([]float64, n)
    // Обратная подстановка: решаем U*x = y
    for i := n - 1; i >= 0; i-- {
        sum := 0.0
        for j := i + 1; j < n; j++ {
            sum += U[i][j] * x[j]
        }
        if U[i][i] == 0 {
            panic("LUSolve: zero diagonal element")
        }
        x[i] = (y[i] - sum) / U[i][i]
    }
    return x
}

// CholeskyDecomposition выполняет разложение Холецкого для симметричной положительно определённой матрицы A (в формате CSR).
// Возвращает нижнюю треугольную матрицу L (dense), такую что A = L * Lᵀ.
func CholeskyDecomposition(A *SparseMatrix) [][]float64 {
    dense := CSRToDense(A)
    n := len(dense)
    L := make([][]float64, n)
    for i := 0; i < n; i++ {
        L[i] = make([]float64, n)
    }
    for i := 0; i < n; i++ {
        for j := 0; j <= i; j++ {
            sum := 0.0
            for k := 0; k < j; k++ {
                sum += L[i][k] * L[j][k]
            }
            if i == j {
                val := dense[i][i] - sum
                if val <= 0 {
                    panic("CholeskyDecomposition: matrix is not positive definite")
                }
                L[i][j] = math.Sqrt(val)
            } else {
                L[i][j] = (dense[i][j] - sum) / L[j][j]
            }
        }
    }
    return L
}

// CholeskySolve решает систему Ax = b, используя разложение Холецкого (A = L * Lᵀ).
func CholeskySolve(L [][]float64, b []float64) []float64 {
    n := len(b)
    y := make([]float64, n)
    // Прямая подстановка: решаем L*y = b
    for i := 0; i < n; i++ {
        sum := 0.0
        for j := 0; j < i; j++ {
            sum += L[i][j] * y[j]
        }
        y[i] = (b[i] - sum) / L[i][i]
    }
    x := make([]float64, n)
    // Обратная подстановка: решаем Lᵀ*x = y
    for i := n - 1; i >= 0; i-- {
        sum := 0.0
        for j := i + 1; j < n; j++ {
            sum += L[j][i] * x[j]
        }
        x[i] = (y[i] - sum) / L[i][i]
    }
    return x
}

// ConjugateGradient решает систему Ax = b для симметричной положительно определённой матрицы A (в CSR) методом сопряжённых градиентов.
// x0 – начальное приближение, tol – допустимая погрешность, maxIter – максимальное число итераций.
func ConjugateGradient(A *SparseMatrix, b []float64, x0 []float64, tol float64, maxIter int) []float64 {
    n := len(b)
    if len(x0) != n {
        panic("ConjugateGradient: initial guess dimension mismatch")
    }
    x := make([]float64, n)
    copy(x, x0)
    r := make([]float64, n)
    Ax := A.MultiplyVector(x)
    for i := 0; i < n; i++ {
        r[i] = b[i] - Ax[i]
    }
    p := make([]float64, n)
    copy(p, r)
    rsold := dot(r, r)

    for iter := 0; iter < maxIter; iter++ {
        Ap := A.MultiplyVector(p)
        alpha := rsold / dot(p, Ap)
        for i := 0; i < n; i++ {
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
        }
        rsnew := dot(r, r)
        if math.Sqrt(rsnew) < tol {
            break
        }
        for i := 0; i < n; i++ {
            p[i] = r[i] + (rsnew/rsold)*p[i]
        }
        rsold = rsnew
    }
    return x
}

// dot вычисляет скалярное произведение двух векторов.
func dot(a, b []float64) float64 {
    if len(a) != len(b) {
        panic("dot: dimension mismatch")
    }
    sum := 0.0
    for i := 0; i < len(a); i++ {
        sum += a[i] * b[i]
    }
    return sum
}


/*
// AMDOrdering вычисляет перестановку для минимизации fill-in
// для симметричной матрицы, представленной в формате CSC.
// Это упрощённая реализация, подходящая для небольших матриц.
func AMDOrdering(csc *SparseMatrixCSC) []int {
	n := csc.Cols // Предполагаем, что матрица квадратная (Rows == Cols)
	// Строим для каждой вершины множество соседей (исходя из паттерна ненулевых элементов)
	neighbors := make([]map[int]struct{}, n)
	for i := 0; i < n; i++ {
		neighbors[i] = make(map[int]struct{})
	}
	// Обходим столбцы CSC: для каждой ненулевой A(i,j) (i != j) добавляем связь и в i, и в j.
	for j := 0; j < n; j++ {
		for idx := csc.ColPtr[j]; idx < csc.ColPtr[j+1]; idx++ {
			i := csc.RowIndex[idx]
			if i != j {
				neighbors[i][j] = struct{}{}
				neighbors[j][i] = struct{}{}
			}
		}
	}
	// Инициализируем степени
	degree := make([]int, n)
	for i := 0; i < n; i++ {
		degree[i] = len(neighbors[i])
	}
	active := make([]bool, n)
	for i := 0; i < n; i++ {
		active[i] = true
	}
	ordering := make([]int, 0, n)

	// Итеративно выбираем активную вершину с минимальной степенью
	for count := 0; count < n; count++ {
		minDeg := math.MaxInt32
		minNode := -1
		for i := 0; i < n; i++ {
			if active[i] && degree[i] < minDeg {
				minDeg = degree[i]
				minNode = i
			}
		}
		if minNode == -1 {
			break
		}
		ordering = append(ordering, minNode)
		active[minNode] = false

		// Обновляем соседей: для всех активных соседей добавляем связи между собой (fill-in)
		nbList := []int{}
		for nb := range neighbors[minNode] {
			if active[nb] {
				nbList = append(nbList, nb)
			}
		}
		for i := 0; i < len(nbList); i++ {
			for j := i + 1; j < len(nbList); j++ {
				a, b := nbList[i], nbList[j]
				if _, ok := neighbors[a][b]; !ok {
					neighbors[a][b] = struct{}{}
					neighbors[b][a] = struct{}{}
					degree[a]++
					degree[b]++
				}
			}
		}
		// Удаляем minNode из соседских списков
		for nb := range neighbors[minNode] {
			if active[nb] {
				delete(neighbors[nb], minNode)
				degree[nb]--
			}
		}
		neighbors[minNode] = nil
	}
	return ordering
}

// NestedDissectionOrdering возвращает перестановку для матрицы на основе Nested Dissection.
// Это упрощённая заглушка; в реальных системах для Nested Dissection часто используют специализированные библиотеки.
func NestedDissectionOrdering(csc *SparseMatrixCSC) []int {
	n := csc.Cols
	ordering := make([]int, n)
	// В данной заглушке возвращается тривиальная перестановка (identity ordering).
	for i := 0; i < n; i++ {
		ordering[i] = i
	}
	return ordering
}
*/


// AMDOrdering вычисляет перестановку для минимизации fill-in для симметричной матрицы в формате CSC.
func AMDOrdering(csc *SparseMatrixCSC) []int {
	n := csc.Cols
	neighbors := make([]map[int]struct{}, n)
	for i := 0; i < n; i++ {
		neighbors[i] = make(map[int]struct{})
	}
	// Построение симметричного графа: для каждой ненулевой A(i,j) (i != j) добавляем ребро между i и j.
	for j := 0; j < n; j++ {
		for idx := csc.ColPtr[j]; idx < csc.ColPtr[j+1]; idx++ {
			i := csc.RowIndex[idx]
			if i != j {
				neighbors[i][j] = struct{}{}
				neighbors[j][i] = struct{}{}
			}
		}
	}
	// Инициализация степеней вершин.
	degree := make([]int, n)
	for i := 0; i < n; i++ {
		degree[i] = len(neighbors[i])
	}
	active := make([]bool, n)
	for i := 0; i < n; i++ {
		active[i] = true
	}
	ordering := make([]int, 0, n)
	// Итеративное исключение вершин.
	for count := 0; count < n; count++ {
		minDeg := math.MaxInt32
		minNode := -1
		for i := 0; i < n; i++ {
			if active[i] && degree[i] < minDeg {
				minDeg = degree[i]
				minNode = i
			}
		}
		if minNode == -1 {
			break
		}
		ordering = append(ordering, minNode)
		active[minNode] = false

		// Собираем список активных соседей.
		nbList := []int{}
		for nb := range neighbors[minNode] {
			if active[nb] {
				nbList = append(nbList, nb)
			}
		}
		// Для каждой пары соседей, если между ними отсутствует ребро, добавляем его (fill-in) и обновляем степени.
		for i := 0; i < len(nbList); i++ {
			for j := i + 1; j < len(nbList); j++ {
				a := nbList[i]
				b := nbList[j]
				if _, ok := neighbors[a][b]; !ok {
					neighbors[a][b] = struct{}{}
					neighbors[b][a] = struct{}{}
					degree[a]++
					degree[b]++
				}
			}
		}
		// Удаляем minNode из списков соседей.
		for nb := range neighbors[minNode] {
			if active[nb] {
				if _, ok := neighbors[nb][minNode]; ok {
					delete(neighbors[nb], minNode)
					degree[nb]--
				}
			}
		}
		neighbors[minNode] = nil
	}
	return ordering
}

// NestedDissectionOrdering вычисляет перестановку для матрицы в формате CSC методом Nested Dissection.
func NestedDissectionOrdering(csc *SparseMatrixCSC) []int {
	n := csc.Cols
	// Строим симметричный граф.
	graph := make([][]int, n)
	for i := 0; i < n; i++ {
		graph[i] = []int{}
	}
	for j := 0; j < n; j++ {
		for idx := csc.ColPtr[j]; idx < csc.ColPtr[j+1]; idx++ {
			i := csc.RowIndex[idx]
			if i != j {
				graph[i] = append(graph[i], j)
				graph[j] = append(graph[j], i)
			}
		}
	}
	// Удаляем дубликаты.
	for i := 0; i < n; i++ {
		m := make(map[int]bool)
		for _, nb := range graph[i] {
			m[nb] = true
		}
		newList := []int{}
		for nb := range m {
			newList = append(newList, nb)
		}
		graph[i] = newList
	}
	used := make([]bool, n)
	return nestedDissectionRecursive(graph, used)
}

// nestedDissectionRecursive выполняет рекурсивное разбиение графа с использованием Nested Dissection.
func nestedDissectionRecursive(graph [][]int, used []bool) []int {
	n := len(graph)
	var nodes []int
	for i := 0; i < n; i++ {
		if !used[i] {
			nodes = append(nodes, i)
		}
	}
	// Базовый случай: если узлов мало, возвращаем их в тривиальном порядке.
	if len(nodes) <= 2 {
		return nodes
	}
	// Выбираем сепаратор: упрощённо выбираем узел с максимальной степенью.
	separator := nodes[0]
	maxDeg := -1
	for _, node := range nodes {
		deg := 0
		for _, nb := range graph[node] {
			if !used[nb] {
				deg++
			}
		}
		if deg > maxDeg {
			maxDeg = deg
			separator = node
		}
	}
	used[separator] = true
	// Находим компоненты связности в подграфе оставшихся узлов.
	comps := connectedComponents(graph, used)
	var ordering []int
	for _, comp := range comps {
		subOrdering := nestedDissectionRecursiveSub(graph, comp)
		ordering = append(ordering, subOrdering...)
	}
	ordering = append(ordering, separator)
	return ordering
}

// nestedDissectionRecursiveSub выполняет рекурсивное упорядочивание для подмножества узлов.
func nestedDissectionRecursiveSub(graph [][]int, comp []int) []int {
	if len(comp) <= 2 {
		return comp
	}
	compSet := make(map[int]bool)
	for _, v := range comp {
		compSet[v] = true
	}
	separator := comp[0]
	maxDeg := -1
	for _, node := range comp {
		deg := 0
		for _, nb := range graph[node] {
			if compSet[nb] {
				deg++
			}
		}
		if deg > maxDeg {
			maxDeg = deg
			separator = node
		}
	}
	var compWithout []int
	for _, v := range comp {
		if v != separator {
			compWithout = append(compWithout, v)
		}
	}
	subComps := connectedComponentsSub(graph, compWithout)
	var ordering []int
	for _, sub := range subComps {
		subOrdering := nestedDissectionRecursiveSub(graph, sub)
		ordering = append(ordering, subOrdering...)
	}
	ordering = append(ordering, separator)
	return ordering
}

// connectedComponents возвращает компоненты связности для узлов, не помеченных как used.
func connectedComponents(graph [][]int, used []bool) [][]int {
	var nodes []int
	for i := 0; i < len(graph); i++ {
		if !used[i] {
			nodes = append(nodes, i)
		}
	}
	return connectedComponentsSub(graph, nodes)
}

// connectedComponentsSub возвращает компоненты связности для заданного набора узлов.
func connectedComponentsSub(graph [][]int, nodes []int) [][]int {
	comp := [][]int{}
	visited := make(map[int]bool)
	compSet := make(map[int]bool)
	for _, v := range nodes {
		compSet[v] = true
	}
	for _, v := range nodes {
		if !visited[v] {
			queue := []int{v}
			current := []int{}
			visited[v] = true
			for len(queue) > 0 {
				cur := queue[0]
				queue = queue[1:]
				current = append(current, cur)
				for _, nb := range graph[cur] {
					if compSet[nb] && !visited[nb] {
						visited[nb] = true
						queue = append(queue, nb)
					}
				}
			}
			comp = append(comp, current)
		}
	}
	return comp
}
