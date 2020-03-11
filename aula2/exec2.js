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
  if (v > 0) return 2
  else if (v < 0) return 1
  else if (v == 0) return 0
}

// Linear activation function
const linear = (v) => {
  if (v >= 0.5) return 2
  else if (v <= -0.5) return 1
  else if (v > -0.5 && v < 0.5) return v
}

// Sigmoid activation function
const sigmoid = (v) => {
  return mathjs.tanh(v)
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

fs.createReadStream('Aula2-exec2.csv')
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
      const weights = [Math.random() * 0.56, Math.random() * 1.30, Math.random() * 0.9, Math.random()]
      rows.forEach((row, idx) => {
        const { v1, v2, v3, v4, v5 } = parseValues(row)
        const values = [v1, v2, v3, v4]

        const v = calculateV(values, weights)
        const biasedV = v - 2.5 * 0.5 // bias = 3.5, biasWeight = 0.5
        // Classifying using threshold function
        const thresholdResult = threshold(biasedV)
        const linearResult = linear(biasedV)
        const sigmoidResult = sigmoid(biasedV)

        // Uncomment line below to print results for each row
        // console.log(`Results for row ${idx + 1}: Threshold -`, thresholdResult, 'Linear -', linearResult, 'Sigmoid -', sigmoidResult)

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





