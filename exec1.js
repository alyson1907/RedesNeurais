const fs = require('fs')
const csv = require('csv-parser')

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

// Threshold function
const threshold = (v) => {
  if (v >= 2) return 2
  else if (v < 2) return 1
}

let bestWeights
let bestAccuracy = 0

const rows = []

// Starting computation
fs.createReadStream('Aula2-exec1.csv')
  .pipe(csv())
  .on('data', row => {
    // Reading data from CSV
    rows.push(row)
  })
  .on('end', () => {
    for (let i = 0; i < 700; i++) {
      let rightGuesses = 0
      const weights = [Math.random(), Math.random() * 10]
      rows.forEach(row => {
        const { v1, v2, v3 } = parseValues(row)
        const values = [v1, v2]

        const v = calculateV(values, weights)
        // Classifying using threshold function
        const result = threshold(v)

        if (guessedRight(result, v3)) {
          rightGuesses++
        }

        const currentAccuracy = rightGuesses / rows.length
        if (currentAccuracy >= bestAccuracy) {
          bestAccuracy = currentAccuracy
          bestWeights = weights
        }
      })
    }
    console.log(`Best accuracy (${bestAccuracy}) for threshold function was found with weights [${bestWeights}]`)

  })





