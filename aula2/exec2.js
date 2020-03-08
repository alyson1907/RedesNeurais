/****************************************************************************** 
**  Disciplina: SCC0270 - Redes Neurais e Aprendizado Profundo
**  Nome: Alyson Matheus Maruyama Nascimento
**  nUSP: 8532269
*******************************************************************************/

const fs = require('fs')
const csv = require('csv-parser')
const mathjs = require('mathjs')

const parseValues = ({ V1, V2, V3, V4, V5 }) => {
  return {
    v1: parseFloat(V1),
    v2: parseFloat(V2),
    v3: parseInt(V3),
    v4: parseFloat(V4),
    v5: parseInt(V5)
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

// Threshold activation function
const threshold = (v) => {
  if (v >= 2) return 2
  else if (v < 2) return 1
}

// Linear activation function
const linear = (v) => {
  const calculated = 0.1 * v
  if (calculated >= 2) return 2
  else if (calculated < 2) return 1
}

// Sigmoid activation function
const sigmoid = (v) => {
  const calculated = 1 / (1 + mathjs.exp(-v))
  if (calculated >= 2) return 2
  else if (calculated < 2) return 1
}

let bestWeights = {
  threshold: [],
  linear: [],
  sigmoid: []
}

let bestAccuracy = {
  threshold: 0,
  linear: 0,
  sigmoid: 0
}

const rows = []

fs.createReadStream('Aula2-exec1.csv')
  .pipe(csv())
  .on('data', row => {
    // Reading data from CSV
    rows.push(row)
  })
  .on('end', () => {
    for (let i = 0; i < 700; i++) {
      let rightGuesses = {
        threshold: 0,
        linear: 0,
        sigmoid: 0
      }
      const weights = [Math.random() * 7, Math.random() * 3, Math.random() * 4, Math.random()]
      rows.forEach(row => {
        const { v1, v2, v3, v4, v5 } = parseValues(row)
        const values = [v1, v2, v3, v4]

        const v = calculateV(values, weights)
        // Classifying using threshold function
        const thresholdResult = threshold(v)
        const linearResult = linear(v)
        const sigmoidResult = sigmoid(v)

        // counting right classifications
        if (guessedRight(thresholdResult, v3)) {
          rightGuesses.threshold += 1
        }
        if (guessedRight(linearResult, v3)) {
          rightGuesses.linear += 1
        }
        if (guessedRight(sigmoidResult, v3)) {
          rightGuesses.sigmoid += 1
        }
      })

      Object.entries(rightGuesses).forEach(([ functionName, rightGuesses ]) => {
        const currentAccuracy = rightGuesses / rows.length
        if (currentAccuracy >= bestAccuracy[functionName]) {
          bestAccuracy[functionName] = currentAccuracy
          bestWeights[functionName] = weights
        }
      })
      
    }
    console.log(`Best accuracy (${bestAccuracy.threshold}) for threshold function was found with weights [${bestWeights.threshold}]`)
    console.log(`Best accuracy (${bestAccuracy.linear}) for linear function was found with weights [${bestWeights.linear}]`)
    console.log(`Best accuracy (${bestAccuracy.sigmoid}) for sigmoid function was found with weights [${bestWeights.sigmoid}]`)
  })





