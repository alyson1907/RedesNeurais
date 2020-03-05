const fs = require('fs')
const csv = require('csv-parser')

let totalNumberOfRows = 0
let rightGuesses = 0

const parseValues = ({ V1, V2, V3 }) => {
  return {
    v1: parseFloat(V1),
    v2: parseFloat(V2),
    v3: parseInt(V3),
  }
}

// We are considering bias = 0
const calculateV = (valuesArray, weightsArray) => {
  return valuesArray.reduce((accumulator, currentValue, idx) => {
    accumulator += currentValue * weightsArray[idx]
    return accumulator
  }, 0)
}

const guessedRight = (result, expectedValue) => {
  if (result == expectedValue) return true
  return false
}

const threshold = (v1, v2) => {
  // Input values
  const values = [v1, v2]
  const weights = [4, 2]
  // Calculating output values
  const v = calculateV(values, weights)
  
  // Threshold function
  if (v >= 2) return 2
  else if (v < 2) return 1
}

fs.createReadStream('Aula2-exec1.csv')
  .pipe(csv())
  .on('data', row => {
    totalNumberOfRows++
    const { v1, v2, v3 } = parseValues(row)
    const result = threshold(v1, v2)
    if (guessedRight(result, v3)) {
      rightGuesses++
    }
  })
  .on('end', () => {
    console.log(`Total Number of rows (input):`, totalNumberOfRows)
    console.log(`Number of correct classifications:`, rightGuesses)
    console.log(`Accuracy:`, rightGuesses/totalNumberOfRows)
  })
  
