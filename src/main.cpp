#include <iostream>
#include <vector>
#include <string>

#include <cmath>
#include <random>
#include <algorithm>

#include <map>
#include <iomanip>
#include <numeric>

// AI systems were used for logic construction, manual confirmation from a human (me) has been conducted
// all C++ code written in this project is entirely from human fingers typing on a keyboard
// this is a learning journey, expect errors perhaps.

// for structs and classes, we will use PascalCase
// for metadata, we will use camelCase
// variables will be readable enough to facilitate info

struct DenseMatrix { 
    // 2D matrix abstraction backed by a contiguous 1D memory buffer through a vector
    
    std::int32_t numberOfRows;
    std::int32_t numberOfColumns;

    std::vector<double> dataElements; // should help in processor cache utilization

    DenseMatrix(std::int32_t rows, std::int32_t columns)
        : numberOfRows(rows), numberOfColumns(columns), dataElements(rows * columns, 0.0) 
        {}

    // the "neurons" are the columns of matrices

    // each row  has numberOfColumns (total columns basically) and total number of rows
    // we will hand the starting position of the row first, then to the element we want
    // (rowIndex * numberOfColumns) + columnIndex
    // row 0 = [0]
    // row 1 = [numberOfColumns]
    // row 2 = [2 * numberOfColumns]
    // row 67 = [67 * numberOfColumns]
    
    // numberOfColumns = 4;
    // rowIndex = 2;
    // columnIndex = 1;

    // [0 1 2 3 4 5 6 7 8 9 10 11]
    //                    ^ - want this 

    // 0 indexing almost hurt me here because im used to starting it with 1 
    // row 0 -> [ 0  1  2  3 ]
    // row 1 -> [ 4  5  6  7 ]
    // row 2 -> [ 8  9  10 11]
    //               ^ - (2,1) 
    // index = (2 * 4) + 1 = 9

    // mutable [RW accessor] pass by addr
    double &getElement(std::int32_t rowIndex, std::int32_t columnIndex) { 
        return {dataElements[(rowIndex * numberOfColumns) + columnIndex]};
    }

    // read only [R accessor] immutable, no modification
    double getElement(std::int32_t rowIndex, std::int32_t columnIndex) const { 
        return {dataElements[(rowIndex * numberOfColumns) + columnIndex]};
    }


};

// a "neuron" in an NN recieves inputs, sums them up, and produces a token score
// simply "adding" would be a linear operation, no matter how many "layers" you add 
// it behaves as a sungle layer only, we dont want this, it cannot learn complex shapes or patterns.
// this can obviously skew results because real world data is not linear, we have many ways of doing stuff
// such as languages and decisions to put, theyre not a linear straight line
// so to solv this, we will add "acitivation functions"

// --- SOFTMAX ---

// an ACTIVATION FUNC is any function that:
// - takes pre-activation values (logits)
// - applies some non-linear transforation to those values
// - produces the post activation output 

// SOFTMAX - convert raw, unnormalized output scores (logits) into a probability distribution
// now this is an ACTIVATION FUNCTION because

// our neural network (and also in general) will output in the form of signed "logits" aka raw scores
// (such as 6.7, 4.20, 0.69 (im sorry))
// by themselves we cant really understand which word to predict next because the scale is arbitrary
// we will write a SOFTMAX fx which will convert these logits into probability distribution
// so we will force all outputs after operation to be unsigned [+ive] between the scale of 0.0 <--> 1.0 (percentile!!!)
// we will also ensure that their sum equalizes to "1.0" 
// logits = [67.0, 1.0, 0.1]
// after softmax: [0.699, 0.242, 0.099]

// s1: make everything positive and exaggerate differences using e^<x>
// suppose 3.0, 1.0, 0.1 (10^-1)
// e^3.0 = 20.08 
// e^1.0 = 2.71 
// e^0.1 = 1.10
// this "gap" makes it more "confident" in its best choice
// ---
// s2: we force it to sum to ONLY 1
//     the process of dividing the exponential of each input value (logit) by the sum of all exponential inputs 
//     (this is called "normalization")
// 20.08 + 2.71 + 1.10 = 23.89
// now, we divide each invidiual exponent by this sum!
// 20.08/23.89 = 0.84 [84%]
// 2.71/23.89  = .011 [11%]
// 1.1/23.89 = 0.05 [5%]
// and would you look at that, 84 + 11 + 5 = 84 + 16 = 100% = ONE!!!

// now this is non-linear because it involves exponential and division operations, rather than simple linear scaling 
// one more thing to note is that this is NOT random output generator, no, its mathematically precise because of said operations
// (does that mean random.randint is too? surely we havent cracked "randomness" right?)

// in math representation and learning: 
// logitVector = [x1, x2, x3, ... xN] (N = number of "classes")
// so now, the probability of P[index] = (e^(x[num]) / sumOf(exponentiatedScores))
// if the raw scores/logits ae very large (1000+), 
// we can subtract a constant (maxElement) from the value such that we get an unharmed result
// Pi ​= exi​−C​/∑j=1N​exj​−C (i dont know how to represent in latex)

std::vector<double> softmax(const std::vector<double> &logitVector) {
    std::vector<double> probabilitiesVector(logitVector.size()); // store normalized probabilities here

    double maxLogitNumeric {*std::max_element(logitVector.begin(), logitVector.end())};
    double sumOfExponentials {0.0}; // zero cuz to adhere additive identity which nomalization follows.

    // PASS 1 - accumulate sum
    for (size_t inx {0}; inx < logitVector.size(); inx += 1) {
        probabilitiesVector[inx] = std::exp(logitVector[inx] - maxLogitNumeric); 
        sumOfExponentials += probabilitiesVector[inx];
    }

    // PASS 2 - normalization
    for (size_t inx {0}; inx < probabilitiesVector.size(); inx += 1) {
        probabilitiesVector[inx] /= sumOfExponentials;
    }

    return probabilitiesVector;
}

// --- ReLU ---

// we will use Rectified Linear Unit (ReLU)
// ReLU is for hidden layers
// softmax we did prior is for the final layer only
// therefore, we use ReLU before softmaxing

// reLU: f(x) = max(0, x)
// f(x) = {x if x > 0, 0 if x <= 0}

// if (PRE_activationVector[inx] > 0.0) {
//    activatedValues[inx] = PRE_activationVector[inx]; // keep the positive value
// } else {
//    activatedValues[inx] = 0.0; // force negative values to 0
// }

// derivative: f'(x) = {1 if x > 0, 0 if x <= 0} (never thought university would come handy like this
// spoilers: this is used in backpropogation apparently
// to train the network of these "neurons" we need to change the "gradient" of the derivative function

std::vector<double> ReLU(const std::vector<double> &PRE_activationVector) {
    std::vector<double> activatedValues(PRE_activationVector.size());

    for (size_t inx {0}; inx < PRE_activationVector.size(); inx += 1) {
        activatedValues[inx] = std::max(0.0, PRE_activationVector[inx]);
    }

    return activatedValues;
}

// --- RANDOM_INIT ---

// good time to introduce you to "Symmetry Problem"
// in a neural network, if the neurons in ANY same layer have the same identical weights and biases
// it will cause them to recieve same inputs and thus produce same outputs
// we dont want this, prior explanations should help you understand now why

// if youve paid attention in your logic circuit classes this variable name should make sense
auto randomInitialization(double fanInput) -> double { 
    static std::mt19937 numericGenerator(42); // seed
    std::uniform_real_distribution<> rangeDistribution((-1.0), (1.0)); // range

    return {rangeDistribution(numericGenerator) * std::sqrt(2.0 / fanInput)}; // scaling factor
}

// -------------------

class CharacterNeuralNetwork {
    private:
        // what we want is that our "hello aris" vector of characters should go through vocabSize
        // it will pass through transformations
        // exit as a probability distribution over the same vocabulary we train it on

        std::int32_t vocabularySize;
        std::int32_t embeddingDensity;
        std::int32_t hiddenDensity;

        DenseMatrix layerOneWeights; // fun-fact: this is called "composition", the instantiation of a class in an extrinsic class
        std::vector<double> layerOneBias;
        // embedding --> hidden via (embeddingDensity x hiddenDensity)
        // matrix multiplication! 
        // each column in this matrix is one "neuron" in the hidden layer

        DenseMatrix layerTwoWeights;
        std::vector<double> layerTwoBias;
        // hidden --> output via (hiddenDensity x vocabSize), MxM!
        // each column in this matrix is one output character (output neurons)

    public:
    
    // allocate main memory (RAM) needed to store connection b/w those 2 layers through list init
        CharacterNeuralNetwork(std::int32_t vocabSize, std::int32_t embeddingDensitySize, std::int32_t hiddenDensitySize)
        : vocabularySize(vocabSize), embeddingDensity(embeddingDensitySize), hiddenDensity(hiddenDensitySize),
          // we will conduct xavier initialization (aka. "glorot initialization") for simplicity
          // which will set the neural network's weights to random values scaled by the number of input neurons and output neurons
          layerOneWeights(embeddingDensitySize, hiddenDensitySize),
          layerOneBias(hiddenDensitySize, static_cast<double>(0.0)),
          layerTwoWeights(hiddenDensitySize, vocabSize),
          layerTwoBias(vocabSize, static_cast<double>(0.0)) 
        {
            for (auto &value : layerOneWeights.dataElements) { value = randomInitialization(embeddingDensity); }
            for (auto &value : layerTwoWeights.dataElements) { value = randomInitialization(hiddenDensity); } 
        }

        // in order to "teach" the AI, it first needs to retain info its been taught/came at
        // these are aka "intemediate values"
        // onto forward pass, every layer of passage = forward pass
        struct ForwardPropogation {
            std::vector<double> hidden; 
            std::vector<double> outputLogits;
            std::vector<double> finalProbabilities;
        };

        ForwardPropogation runForwardPass(const std::vector<double> &embedding) {
            ForwardPropogation result;

            result.hidden.resize(hiddenDensity);
            // result.outputLogits.resize(vocabularySize);
            // sum = bias + (input * weight)
            // basic linear transformation for a single neuron in an NN 
            // using beloved MxMs! (matrix multiplication!) 

            for (size_t inx {0}; inx < hiddenDensity; inx += 1) {

                double accumulateSum1 {layerOneBias[inx]};
                for (size_t ext {0}; ext < embeddingDensity; ext += 1) {
                    accumulateSum1 += embedding[ext] * layerOneWeights.getElement(ext, inx);
                }

                result.hidden[inx] = std::max(0.0, accumulateSum1); // ReLU operation, will skip negative values
                // this should give us an idea about the active neurons in a pass
            }

            result.outputLogits.resize(vocabularySize);
            for (size_t inx {0}; inx < vocabularySize; inx += 1) {
                
                double accumulateSum2 {layerTwoBias[inx]};
                for (size_t ext {0}; ext < hiddenDensity; ext += 1) {
                    accumulateSum2 += result.hidden[ext] * layerTwoWeights.getElement(ext, inx);
                } 

                result.outputLogits[inx] = accumulateSum2;
            }

            // we return the final activation function output as distributed probability
            result.finalProbabilities = softmax(result.outputLogits);
            return result;
        }

        auto singularTrainingStep( // backpropogation + gradient descent
            const std::vector<double> &inputEmbedding, std::int32_t targetTokenID, double learningRate
        ) -> void {
            // s1 - forward pass
            ForwardPropogation forwardOutput {runForwardPass(inputEmbedding)};

            // s2 - backward pass
            // we will compute gradients via chain rule
            // for softmax + cross-entropy,
            // the gradient becomes gradLoss/gradLogits = predicted probability - oneHot

            // BACKPROPOGATION
            // suppose it predicted 'x' with 0.8 = 80%
            // if 'x' was the tokenID/correct, then we get the error (0.8 - 1 = (-0.2)) (need to push it up because its correct)
            // if it pedicted 'y' at 0.2 = 20%, then the error becomes (0.2 -0 = 0.2) (need to push prob down)

            // s3 - first we calc the error exactly, we backpropogate to hidden layer
            // chain rule: hidden layer = (outputErr * layerWeightTranspose) * ReLU derivative

            // derivative of the Loss Function with respect to the inputs
            // why is this so confusing?
            // partial derivative of loss = d(loss) 
            // partial derivative of logit/input = d(input)
            // d(loss)/d(input) = error change "difference" WRT change in raw model output prior to activation
            // wait, its partial derivative only, right?
            std::vector<double> outputErrorSignal {forwardOutput.finalProbabilities};
            outputErrorSignal[targetTokenID] -= (1.0); // cross-entropy dx, subtract 1 from "correct" character's probability
            std::vector<double> hiddenErrorSignal(hiddenDensity, (0.0));

            // we will propogate the error to the hidden neuron layer
            for (size_t neuronIndex {0}; neuronIndex < hiddenDensity; neuronIndex += 1) {
                double errorSum {0.0};

                for (size_t outputIndex {0}; outputIndex < vocabularySize; outputIndex += 1) {
                    errorSum += layerTwoWeights.getElement(neuronIndex, outputIndex) * outputErrorSignal[outputIndex]; 
                }

                // we will apply ReLU derivative now to follow chain rule
                // if the neuron was "negative" (<=0) during forward pass, it gets no blame (error is 0)
                // if it was positive (>0), it takes the full error
                double wasNeuronActive {(forwardOutput.hidden[neuronIndex] > 0.0) ? 1.0 : 0.0};
                hiddenErrorSignal[neuronIndex] = errorSum * wasNeuronActive;
            }

            // s4 - updating of layer (2) weights and biases = gradient descent
            for (size_t neuronIndex {0}; neuronIndex < hiddenDensity; neuronIndex += 1) {
                for (size_t outputIndex {0}; outputIndex < vocabularySize; outputIndex += 1) {
                    double currentWeight = layerTwoWeights.getElement(neuronIndex, outputIndex);
                    
                    // gradient = error * input
                    // gradient = outputError * hiddenNeuronVal 
                    double gradient {(outputErrorSignal[outputIndex] * forwardOutput.hidden[neuronIndex])};
                    layerTwoWeights.getElement(neuronIndex, outputIndex) = (currentWeight - (learningRate * gradient));
                }
            }

            // update layer 1 weights and biases
            for (size_t outputIndex {0}; outputIndex < vocabularySize; outputIndex += 1) {
                layerTwoBias[outputIndex] -= learningRate * outputErrorSignal[outputIndex];
            }

            for (size_t inputIndex {0}; inputIndex < embeddingDensity; inputIndex += 1) {
                for (size_t neuronIndex {0}; neuronIndex < hiddenDensity; neuronIndex += 1) {
                    double currentWeight = layerOneWeights.getElement(inputIndex, neuronIndex);
                    
                    // gradient {hidden Error * input value};
                    double gradient = hiddenErrorSignal[neuronIndex] * inputEmbedding[inputIndex];
                    layerOneWeights.getElement(inputIndex, neuronIndex) = (currentWeight - (learningRate * gradient));
                }
            }

            for (size_t neuronIndex {0}; neuronIndex < hiddenDensity; neuronIndex += 1) {
                layerOneBias[neuronIndex] -= learningRate * hiddenErrorSignal[neuronIndex];
            }
        }
};

std::int32_t main(std::int32_t argc, char *argv[]) { 

    return static_cast<std::int32_t>(NULL); // just zero 
}
