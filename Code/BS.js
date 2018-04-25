/*
    Extension of William Basener's JavaScript by Tom English.
    
    William Basener holds the copyrights for the original code, which I use
    with his permission.
*/

//# var AnimateId; unused

//# The follow are the parameters input via HTML in Basener's
//# webpage. Only the first of them is global in Basener's script.

const process = require('process');

PctBeneficial = process.argv[2]; // "Percentage of mutations that are beneficial"
mt = process.argv[3];            // "Mutation Distribution Type"
PopSize = process.argv[4];       // "Population Size" is Finite or Infinite
numYears = process.argv[5];      // "Number of years" in the evolutionary process
numIncrements = process.argv[6]; // "Number of Discrete Population Fitness Values"
output_path = process.argv[7];



function Gamma(Z) {
    with(Math) {
        var S = 1 + 76.18009173 / Z - 86.50532033 / (Z + 1) + 24.01409822 / (Z + 2) - 1.231739516 / (Z + 3) + .00120858003 / (Z + 4) - .00000536382 / (Z + 5);
        var G = exp((Z - .5) * log(Z + 4.5) - (Z + 4.5) + log(S * 2.50662827465));
    }
    return G
}



function mutationProb(mDiff, mDelta, mt) {
    if (mt == "Gaussian") {
        var stdevMutation = /* 0.0005 */ 0.002;  //# 0.002 in BS article
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
        if (mDiff == 0) mDiff = -mDelta;
        var sBarBeneficial = 0.001;
        var sBarDeleterious = 0.001;
        var aBeneficial = 0.5;
        var aDeleterious = 0.5;
        var bBeneficial = aBeneficial / sBarBeneficial;
        var bDeleterious = aDeleterious / sBarDeleterious;
        if (mDiff > 0) f = (PctBeneficial) * Math.pow(bBeneficial, aBeneficial) * Math.pow(mDiff, aBeneficial - 1) * Math.exp(-bBeneficial * mDiff) / Gamma(aBeneficial);
        if (mDiff < 0) f = (1 - PctBeneficial) * Math.pow(bDeleterious, aDeleterious) * Math.pow(Math.abs(mDiff), aDeleterious - 1) * Math.exp(-bDeleterious * Math.abs(mDiff)) / Gamma(aDeleterious);
        f = f * mDelta;
    }
    if (mt == "None" || mt == "NoneExact") {
        f = 0;
        if (mDiff == 0) f = 1;
    }
    return f;
}



function runSimulation() {
    
    //# The parameters that Basener set via HTML are now global.
    
    console.log('Working...');
    
    //# Removed code for setting the parameters via HTML.
    //# var maxPopulationSize = 10 ^ 9; unused
    var mean = 0.044;
    var stdev = 0.005;
    var fitnessRange = [-0.1, 0.15];
    var numStDev = 11.2;
    var deathRate = 0.1;
    var meanFitness = new Array(numYears);
    var varianceFitness = new Array(numYears);
    var P = new Array(numIncrements);
    
    //# The following are unused.
    //# var MeanROC = new Array(numYears);
    //# var MVRatio = new Array(numYears);
    //# var MVDiff = new Array(numYears);
    //# var yearVariable = new Array(numYears);
    
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
    //# Correct error: last iteration assigns to memory out of bounds
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
    
    s = 0;
    for (i = 0; i < numIncrements; i++) {
        s = s + P[i]
    }
    
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
    
    for (t = 1; t < numYears; t++) {
        s = 0;
        meanFitness[t] = 0;
        varianceFitness[t] = 0;
        for (i = 0; i < numIncrements; i++) {
            Psolution[t][i] = Psolution[t - 1][i];
            for (j = 0; j < numIncrements; j++) {
                Psolution[t][i] = Psolution[t][i] + Psolution[t - 1][j] * MP[i][j];
            }
            Psolution[t][i] = Psolution[t][i] - deathRate * Psolution[t - 1][i];
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

    var probs = new Array(2 * numIncrements - 1);
    var n = numIncrements - 1;
    for (var i = 0; i < b.length; i++) {
        probs[n-i] = mutationProb(-b[i], mDelta, mt);
        probs[n+i] = mutationProb(b[i], mDelta, mt);
    }
    
    // Added to write results to disk
    const results = {'PctBeneficial' : PctBeneficial, // percent_beneficial
                     'mt' : mt,                       // mutation_type
                     'PopSize' : PopSize,             //population_size
                     'numYears' : numYears,
                     'numIncrements' : numIncrements, // n_rates
                     'b' : b,                         //birth_rates
                     'Psolution' : Psolution,         // trajectory
                     'm' : m,                         // growth_rates
                     'meanFitness': meanFitness,      // means
                     'varianceFitness' : varianceFitness, // variances
                     'mutation_probs' : probs,        // added by TME
                     'mDelta' : mDelta};              // bin_width
    const fs = require('fs');
    const output = JSON.stringify(results);
    fs.writeFile(output_path, output, 'utf8', function (error) {
        if (error) {
            return console.log(error);
        }
        console.log("... output is in " + output_path);
    });
}

runSimulation();
