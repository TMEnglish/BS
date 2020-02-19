/*
Adaptation of an excerpt of the JavaScript in

    Realistic Mutation-Selection Modeling
    https://people.rit.edu/wfbsma/evolutionary%20dynamics/EvolutionaryModel.html
    Copyright William Basener

to run from the command line, using Node.js.

Adapted by Tom English with permission of William Basener.

Use of this software is restricted to scholarly investigation of

    "The fundamental theorem of natural selection with mutations,"
    Basener, W.F. & Sanford, J.C. J. Math. Biol. (2018) 76: 1589.
    https://doi.org/10.1007/s00285-017-1190-x

Redistribution is prohibited.
*/

// var AnimateId; unused

// The follow are the parameters input via HTML in Basener's webpage.
// Only the first of them is global in Basener's script.

const process = require('process');
const jStat = require('jStat').jStat;

PctBeneficial = process.argv[2]; // "Percentage of mutations that are beneficial"
mt = process.argv[3];            // "Mutation Distribution Type"
PopSize = process.argv[4];       // "Population Size" is Finite or Infinite
numYears = process.argv[5];      // "Number of years" in the evolutionary process
numIncrements = process.argv[6]; // "Number of Discrete Population Fitness Values"
output_path = process.argv[7];


// Copied from the Wikipedia article on Kahan Summation Algorithm

function KahanSum(input) {
    var sum = 0.0;
    var c = 0.0;                 // A running compensation for lost low-order bits.
    for (i = 0; i < input.length; i++) {
        var y = input[i] - c;    // So far, so good: c is zero.
        var t = sum + y;         // Alas, sum is big, y small, so low-order digits of y are lost.
        c = (t - sum) - y;       // (t - sum) cancels the high-order part of y; subtracting y recovers negative (low part of y)
        sum = t;                 // Algebraically, c should always be zero. Beware overly-aggressive optimizing compilers!
    }                            // Next time around, the lost low part will be added to y in a fresh attempt.
    return sum
}



function Gamma(Z) {
    with(Math) {
        var S = 1 + 76.18009173 / Z - 86.50532033 / (Z + 1) + 24.01409822 / (Z + 2) - 1.231739516 / (Z + 3) + .00120858003 / (Z + 4) - .00000536382 / (Z + 5);
        var G = exp((Z - .5) * log(Z + 4.5) - (Z + 4.5) + log(S * 2.50662827465));
    }
    return G
}



function mutationProb(mDiff, mDelta, mt) {
    // mDiff is the difference in fitness of the offspring from the parent
    // mDelta is the spacing of discrete fitnesses in the fitness interval
    // mt is the "mutation distribution type"

    if (mt == "Gaussian") {
        var stdevMutation = /* 0.0005 */ 0.002;  // as reported
        GaussianMultiplicativeTerm = 1 / (stdevMutation * Math.sqrt(2 * Math.PI));
        f = GaussianMultiplicativeTerm * Math.exp(-0.5 * Math.pow((mDiff) / stdevMutation, 2));
        f = f * mDelta;
    }
    if (mt == "Asymmetrical Gaussian") {
        var stdevMutation = 0.001;
        var stdevMutationBeneficial = 0.001;
        if (mDiff > 0) {
            GaussianMultiplicativeTerm = 1 / (stdevMutationBeneficial * Math.sqrt(2 * Math.PI));
            f = ((PctBeneficial) / 0.5) * GaussianMultiplicativeTerm * Math.exp(-0.5 * Math.pow((mDiff) / stdevMutationBeneficial, 2));
        }
        else {
            GaussianMultiplicativeTerm = 1 / (stdevMutation * Math.sqrt(2 * Math.PI));
            f = ((1 - PctBeneficial) / 0.5) * GaussianMultiplicativeTerm * Math.exp(-0.5 * Math.pow((mDiff) / stdevMutation, 2));
        }
        f = f * mDelta;
    }
    if (mt == "Gamma") {
        if (mDiff == 0)      // If difference in fitness is 0,
            mDiff = -mDelta; // change it to the minimum negative difference.
        // Basener considered making the parameters different for the upper
        // and lower tails.
        var sBarBeneficial = 0.001;
        var sBarDeleterious = 0.001;
        var aBeneficial = 0.5;
        var aDeleterious = 0.5;
        var bBeneficial = aBeneficial / sBarBeneficial;
        var bDeleterious = aDeleterious / sBarDeleterious;
        const alpha=0.5;
        const beta=0.5/0.001;
        f = jStat.gamma.pdf(Math.abs(mDiff), alpha, 1 / beta);
        if (mDiff > 0)
            // Simple scaling by PctBeneficial tacitly assumes that the mass
            // assigned to positive effects by the Gamma distribution is 1.
            f *= PctBeneficial;
        if (mDiff < 0) 
            // Simple scaling by 1 - PctBeneficial tacitly assumes that the mass
            // assigned to negative mutation effects by the reflected Gamma 
            // distribution is 1.
            f *= 1 - PctBeneficial;
        f = f * mDelta; // multiply probability density by length of subinterval
    }
    if (mt == "None" || mt == "NoneExact") {
        f = 0;
        if (mDiff == 0) f = 1;
    }
    return f;
}



function runSimulation() {
    
    //# The parameters that Basener set via HTML are now global.
    
    console.log('Initializing.');
    
    // Removed code for setting the parameters via HTML.
    // var maxPopulationSize = 10 ^ 9; unused
    var mean = 0.044;
    var stdev = 0.005;
    var fitnessRange = [-0.1, 0.15];
    var numStDev = 11.2;
    var deathRate = 0.1;
    var meanFitness = new Array(numYears);
    var varianceFitness = new Array(numYears);
    var P = new Array(numIncrements);
    
    // The following are unused.
    // var MeanROC = new Array(numYears);
    // var MVRatio = new Array(numYears);
    // var MVDiff = new Array(numYears);
    // var yearVariable = new Array(numYears);
    
    var minFitness = mean - numStDev * stdev;
    var maxFitness = mean + numStDev * stdev;
    var m = new Array(numIncrements);
    mDelta = (fitnessRange[1] - fitnessRange[0]) / (numIncrements);
    
    m[0] = fitnessRange[0];
    for (i = 1; i < numIncrements; i++) {
        m[i] = fitnessRange[0] + i * mDelta;
    }
    
    var b = new Array(numIncrements);
    for (i = 0; i < numIncrements; i++) {
        b[i] = m[i] + deathRate;
    }
    
    GaussianMultiplicativeTerm = 1 / (stdev * Math.sqrt(2 * Math.PI));
    // Correct error: last iteration assigns to memory out of bounds
    for (i = 0; i < numIncrements /* + 1 */; i++) { 
        P[i] = GaussianMultiplicativeTerm * Math.exp(-0.5 * Math.pow((m[i] - mean) / stdev, 2));
        if (m[i] < minFitness) {
            P[i] = 0
        };
        if (m[i] > maxFitness) {
            P[i] = 0
        };
    }
    
    var Psolution = new Array(numYears);
    var PsolutionForPlot = new Array(numYears);
    for (var t = 0; t < numYears; t++) {
        Psolution[t] = new Array(numIncrements);
        PsolutionForPlot[t] = new Array(numIncrements);
    }
    
    /*
    s = 0;
    for (i = 0; i < numIncrements; i++) {
        s = s + P[i]
    }
    */
    s = KahanSum(P); // changed by TME
    
    var maxPinitial = 0;
    for (var i = 0; i < numIncrements; i++) {
        Psolution[0][i] = P[i] / s;
        PsolutionForPlot[0][i] = P[i] / s;
        maxPinitial = Math.max(maxPinitial, PsolutionForPlot[0][i]);
    }
    
    var MP = new Array(numIncrements);    
    for (i = 0; i < numIncrements; i++) {
        MP[i] = new Array(numIncrements);
        for (j = 0; j < numIncrements; j++) {
            MP[i][j] = b[j] * mutationProb(m[i] - m[j], mDelta, mt);
        }
    }
    
    meanFitness[0] = mean;
    varianceFitness[0] = 0;
    for (var i = 0; i < numIncrements; i++) {
        varianceFitness[0] = varianceFitness[0] + (m[i] - mean) * (m[i] - mean) * Psolution[0][i];
    }

    console.log('Iterating.');

    for (t = 1; t < numYears; t++) {
        s = 0;
        meanFitness[t] = 0;
        varianceFitness[t] = 0;
        for (i = 0; i < numIncrements; i++) {
            /*
            Never initialize an accumulator with a number much larger than
            the remaining summands, as Basener does here:
            
            Psolution[t][i] = Psolution[t - 1][i];
            
            for (j = 0; j < numIncrements; j++) {
                Psolution[t][i] = Psolution[t][i] + Psolution[t - 1][j] * MP[i][j];
            }
            
            Psolution[t][i] = Psolution[t][i] - deathRate * Psolution[t - 1][i];
            
            To improve precision considerably, add the sum of the (small)
            numbers of births to the (large) number of survivors.
            */
            
            var births = jStat.dot(Psolution[t-1], MP[i]);
            Psolution[t][i] = births + (1 - deathRate) * Psolution[t - 1][i];
           
            if (mt == "NoneExact") 
                Psolution[t][i] = Psolution[0][i] * Math.exp(t * m[i]);
            s = s + Psolution[t][i];
        }
        if (PopSize == "Finite") {
            maximumP = Math.max.apply(Math, Psolution[t]);
            for (i = 0; i < numIncrements; i++) {
                Psolution[t][i] = Psolution[t][i] * (Psolution[t][i] > maximumP * 0.000000001);
            }
        }
        for (var i = 0; i < numIncrements; i++) {
            PsolutionForPlot[t][i] = Psolution[t][i] / s;
            meanFitness[t] = meanFitness[t] + m[i] * PsolutionForPlot[t][i];
        }
        for (var i = 0; i < numIncrements; i++) {
            mMinusMean = (m[i] - meanFitness[t]);
            varianceFitness[t] = varianceFitness[t] + mMinusMean * mMinusMean * PsolutionForPlot[t][i];
        }
    }
    
    // Added to calculate the probabilities of mutation effects
    var mutation_probs = new Array(2 * b.length - 1);
    var zero_index = b.length - 1;
    
    for (var i = 0; i < b.length; i++) {
        mutation_probs[zero_index - i] = mutationProb(-b[i], mDelta, mt);
        mutation_probs[zero_index + i] = mutationProb(b[i], mDelta, mt);
    }
    
    // Added to write results to disk
    const results = {'PctBeneficial' : PctBeneficial,     // percent_beneficial
                     'mt' : mt,                           // mutation_type
                     'PopSize' : PopSize,                 // population_size
                     'numYears' : numYears,               // n_years + 1
                     'numIncrements' : numIncrements,     // n_types
                     'b' : b,                             // birth_factors
                     'P' : P,                             // unnormalized init
                     'Psolution' : Psolution,             // trajectory
                     'm' : m,                             // growth_factors
                     'meanFitness': meanFitness,          // means
                     'varianceFitness' : varianceFitness, // variances
                     'mutation_probs' : mutation_probs,   // added
                     'MP' : MP,                           // birthing matrix
                     'mDelta' : mDelta};                  // delta
    
    console.log("Writing output to " + output_path);
    const fs = require('fs');
    const output = JSON.stringify(results);
    fs.writeFile(output_path, output, 'utf8', function (error) {
        if (error) {
            return console.log(error);
        }
    });
    console.log("Done.")
}

runSimulation();
