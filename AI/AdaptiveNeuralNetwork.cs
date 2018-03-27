/* This source code provides an improved implementation of the 
 * Adaptive Resonance Associative Map, proposed by Dr. Ah-Hwee Tan
 * as ART Map extension that displays a faster and more stable 
 * behavior for neuron categorization and prediction.
 * Copyright(C) 2018 Alysson Ribeiro da Silva
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.If not, see<https://www.gnu.org/licenses/>.
 * 
 * If you want to ask me any question, you can send an e-mail to 
 * Alysson.Ribeiro.Silva@gmail.com entitled as "C# ADAPTIVE NEURAL NETWORK"
*/

using System;
using System.Collections.Generic;

namespace TCAI
{
    /// <summary>
    /// Created by Alysson Ribeiro da Silva, the ComplementCodingType specifies which complement will be used when performing prediction or learning
    /// </summary>
    public class ComplementCodingType
    {
        public const int NONE = 0;
        public const int MIRRORED = 1;
        public const int PREDICTION = 2;
        public const int DIRECT_ACCESS = 3;
    }

    /// <summary>
    /// Created by Alysson Ribeiro da Silva, the FieldTypes class is responsible for holding all names to represent filed types
    /// </summary>
    public class FieldTypes
    {
        // Field type variables
        public const int STATE = 0;
        public const int ACTION = 1;
        public const int REWARD = 2;
    }

    /// <summary>
    /// Created by Alysson Ribeiro da Silva, the NeuronLearning class is responsible for holding all neuron learning types for the fuzzy neuron clusters
    /// </summary>
    public class NeuronLearning
    {
        public const int ART_I = 1;
        public const int ART_II = 2;
    }

    /// <summary>
    /// Created by Alysson Ribeiro da Silva, the NeuronActivation class is responsible for holding all the activaton types for the fuzzy neuron clusters
    /// </summary>
    public class NeuronActivation
    {
        public const int ART_I = 1;
        public const int ART_II = 2;
        public const int PROXIMITY = 3;
        public const int ELEMENT = 4;
    }

    /// <summary>
    ///  Created by Alysson Ribeiro da Silva, the NetDescription class is responsible for holding the structural design of the Adaptive Neural Network.
    /// </summary>
    public class NetDescription
    {
        public int[] featuresSizes;
        public bool[] adaptiveVigilanceRaising;
        public bool fuzzyReadout;
        public bool[] activeFields;
        public int[] temperatureOp;
        public int[] learningOp;
        public int[] fieldsClass;
        public double[] learningRate;
        public double[] learningVigilances;
        public double[] performingVigilances;
        public double[] gammas;
        public double[] alphas;
        public double adaptiveVigilanceRate;
    }

    /// <summary>
    ///  Created by Alysson Ribeiro da Silva, the NeuronTemperatureTuple class is responsible for holding the neuron temperature and its index to facilitate the network's
    ///  internal operations.
    /// </summary>
    public class NeuronTemperatureTuple
    {
        // -------------------------------------------------------------------------------------------------------
        public NeuronTemperatureTuple(int neuronIndex, double t)
        {
            this.t = t;
            this.neuronIndex = neuronIndex;
        }
        // -------------------------------------------------------------------------------------------------------
        public double t;
        public int neuronIndex;
        // -------------------------------------------------------------------------------------------------------
    }

    /// <summary>
    ///  Created by Alysson Ribeiro da Silva, the AdaptiveNeuralNetwork class is implements the Adaptive Resonance Associative Map with the composite operations.
    ///  The deployed model enables to use fuzzy ART I, fuzzy ART II, and a proximity metric for neuron cluster categorization.
    ///  This class helps deploying Q-Learning models and Reactive models, based on action masks, to control agents in real-time.
    ///  Moreover, it also posses a perfect miss-match mechanism, and neuron cluster based operations to facilitate its working and to avoid errors when predicting information.
    /// </summary>
    public class AdaptiveNeuralNetwork
    {
        // -------------------------------------------------------------------------------------------------------
        /// <summary>
        /// Creates a new ANN with the specified configuration class
        /// </summary>
        /// <param name="config"></param>
        public AdaptiveNeuralNetwork(NetDescription config)
        {
            this.totalFields = config.featuresSizes.Length;
            this.vigilancesRaising = config.adaptiveVigilanceRaising;
            this.fuzzyReadout = config.fuzzyReadout;
            this.activeFields = config.activeFields;
            this.featuresSizes = config.featuresSizes;
            this.temperatureOp = config.temperatureOp;
            this.learningOp = config.learningOp;
            this.fieldsClass = config.fieldsClass;
            this.learningRate = config.learningRate;
            this.learnVigilances = config.learningVigilances;
            this.performVigilances = config.performingVigilances;
            this.gammas = config.gammas;
            this.alphas = config.alphas;
            this.adaptiveVigilanceRate = config.adaptiveVigilanceRate;

            activity = new double[totalFields][];
            for (int field = 0; field < totalFields; field++)
                activity[field] = new double[featuresSizes[field]];

            inputFields = new double[totalFields][];
            for (int field = 0; field < totalFields; field++)
                inputFields[field] = new double[featuresSizes[field]];

            predictionFields = new double[totalFields][];
            for (int field = 0; field < totalFields; field++)
                predictionFields[field] = new double[featuresSizes[field]];

            activitySum = new double[totalFields];

            neuronsConfidence = new List<double>();
            neurons = new List<double[][]>();

            createNeuron();
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Configures the Q-Learning dynamics parameters 
        /// </summary>
        /// <param name="discount"></param>
        /// <param name="learningRate"></param>
       /* public void configureQLearningHelper(double discount, double learningRate)
        {
            this.qDiscountParameter = discount;
            this.qLearningRate = learningRate;
        }*/
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Configures the reactive model dynamics parameters
        /// </summary>
        /// <param name="erosionRate"></param>
        /// <param name="reinforcementRate"></param>
        /// <param name="decayRate"></param>
        /// <param name="confidenceThreshold"></param>
        /// <param name="prunningThreshold"></param>
        public void configureReactiveHelper(
            double erosionRate,
            double reinforcementRate,
            double decayRate,
            double confidenceThreshold,
            int prunningThreshold)
        {
            // Reactive model helper dynamics variables
            this.neuronErosionRate = erosionRate;
            this.neuronReinforcementRate = reinforcementRate;
            this.neuronDecayRate = decayRate;
            this.neuronConfidenceThreshold = confidenceThreshold;
            this.neuronPrunningThreshold = prunningThreshold;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Inserts a new neuron cluster into the ANN
        /// </summary>
        private void createNeuron()
        {
            double[][] neuron = new double[totalFields][];

            for (int field = 0; field < totalFields; field++)
            {
                neuron[field] = new double[featuresSizes[field]];
                for (int element = 0; element < featuresSizes[field]; element++)
                {
                    neuron[field][element] = 1.0;
                }
            }

            neurons.Add(neuron);
            neuronsConfidence.Add(1.0);
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Set the activity to operate 
        /// </summary>
        /// <param name="stimulus"></param>
        public void setActivity(double[][] stimulus)
        {
            for (int field = 0; field < totalFields; field++)
                Array.Copy(stimulus[field], 0, activity[field], 0, featuresSizes[field]);
        }
        //----------------------------------------------------------------------------------------------------        
        /// <summary>
        /// Reads the current activity
        /// </summary>
        private void calculateActivitySum()
        {
            for (int field = 0; field < totalFields; field++)
            {
                activitySum[field] = 0.0;
                for (int element = 0; element < featuresSizes[field]; element++)
                    activitySum[field] += activity[field][element];
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Calculates the fuzzy ART I categorization measurement 
        /// </summary>
        /// <param name="field"></param>
        /// <param name="neuronField"></param>
        /// <param name="neuron_wAndxSum"></param>
        /// <returns></returns>
        private double ARTI(int field, double[] neuronField, double[] neuron_wAndxSum)
        {
            double xAndwSum = 0.0;
            double wSum = 0.0;
            for (int element = 0; element < featuresSizes[field]; element++)
            {
                xAndwSum += Math.Min(neuronField[element], activity[field][element]);
                wSum += neuronField[element];
            }

            neuron_wAndxSum[field] = xAndwSum;

            double t = (xAndwSum / (alphas[field] + wSum));

            if (t == Double.NaN)
                t = 0.00001;

            return t * gammas[field];
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Calculates the fuzzy ART II categorization measurement
        /// </summary>
        /// <param name="field"></param>
        /// <param name="neuronField"></param>
        /// <param name="neuron_wAndxSum"></param>
        /// <returns></returns>
        private double ARTII(int field, double[] neuronField, double[] neuron_wAndxSum)
        {
            double xAndwSum = 0.0;
            double xDotw = 0.0;
            double wLenght = 0.0;
            double xLenght = 0.0;

            for (int element = 0; element < featuresSizes[field]; element++)
            {
                xAndwSum += Math.Min(neuronField[element], activity[field][element]);
                xDotw += neuronField[element] * activity[field][element];
                wLenght += Math.Pow(neuronField[element], 2.0);
                xLenght += Math.Pow(activity[field][element], 2.0);
            }

            neuron_wAndxSum[field] = xAndwSum;

            wLenght = Math.Sqrt(wLenght);
            xLenght = Math.Sqrt(xLenght);

            double t = xDotw / (alphas[field] + (wLenght * xLenght));

            if (t == Double.NaN)
                t = 0.00001;

            return t * gammas[field];
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Calculates the proximity categorization measurement 
        /// </summary>
        /// <param name="field"></param>
        /// <param name="neuronField"></param>
        /// <param name="neuron_wAndxSum"></param>
        /// <returns></returns>
        private double proximity(int field, double[] neuronField, double[] neuron_wAndxSum)
        {
            double xAndwSum = 0.0;
            double dist = 0.0;

            for (int element = 0; element < featuresSizes[field]; element++)
            {
                xAndwSum += Math.Min(neuronField[element], activity[field][element]);
                dist += Math.Abs(neuronField[element] - activity[field][element]);
            }

            double t = 1.0 / (alphas[field] + dist);
            neuron_wAndxSum[field] = xAndwSum;
            // neuron_wAndxSum[field] = 1.0 - (dist / (double)
            // featuresSizes[field]);

            if (t == Double.NaN)
                t = 0.000001;

            return t * gammas[field];
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Calculates the element by element categorization measurement 
        /// </summary>
        /// <param name="field"></param>
        /// <param name="neuronField"></param>
        /// <param name="neuron_wAndxSum"></param>
        /// <returns></returns>
        private double element(int field, double[] neuronField, double[] neuron_wAndxSum)
        {
            bool elementCheck = true;
            for (int element = 0; element < featuresSizes[field]; element++)
            {
                if (neuronField[element] != activity[field][element])
                {
                    elementCheck = false;
                    break;
                }
            }

            double t = 0.0;
            neuron_wAndxSum[field] = 0.0;

            if (elementCheck)
            {
                t = 1.0;
                neuron_wAndxSum[field] = 1.0;
            }

            return t * gammas[field];
        }
        //----------------------------------------------------------------------------------------------------        
        /// <summary>
        /// Calculates composite operation 
        /// </summary>
        /// <param name="neuron"></param>
        /// <param name="neuron_wAndxSum"></param>
        /// <returns></returns>
        private double calculateTComposite(double[][] neuron, double[] neuron_wAndxSum)
        {
            double t = 0.0;
            for (int field = 0; field < totalFields; field++)
            {
                if (activeFields[field])
                {
                    int op = temperatureOp[field];
                    switch (op)
                    {
                        case NeuronActivation.ART_I:
                            t += ARTI(field, neuron[field], neuron_wAndxSum);
                            break;
                        case NeuronActivation.ART_II:
                            t += ARTII(field, neuron[field], neuron_wAndxSum);
                            break;
                        case NeuronActivation.PROXIMITY:
                            t += proximity(field, neuron[field], neuron_wAndxSum);
                            break;
                        case NeuronActivation.ELEMENT:
                            t += element(field, neuron[field], neuron_wAndxSum);
                            break;
                    }
                }
            }

            return t;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Calculates the match factor between the neuron cluster weights and the received stimulus ones 
        /// </summary>
        /// <param name="field"></param>
        /// <param name="selectedNeuronField"></param>
        /// <param name="wAndxSum"></param>
        /// <returns></returns>
        private double doMatch(int field, double[] selectedNeuronField, double wAndxSum)
        {
            double m_j = wAndxSum / activitySum[field];

            return m_j;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Verifies if a perfect miss match occurred, when the verified stimulus is 
        /// </summary>
        /// <param name="field"></param>
        /// <param name="neuron"></param>
        /// <returns></returns>
        private bool checkPerfectMissmatch(int field, double[][] neuron)
        {
            bool pmm = true;
            for (int i = 0; i < neuron[field].Length; i++)
            {
                if (neuron[field][i] != activity[field][i])
                    pmm = false;
            }

            return pmm;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Stamps the received stimulus into the selected neuron cluster weights with the fuzzy ART I operation 
        /// </summary>
        /// <param name="field"></param>
        /// <param name="selectedNeuron"></param>
        private void stampNeuronARTI(int field, int selectedNeuron)
        {
            double[][] learningNeuron = neurons[selectedNeuron];
            for (int element = 0; element < featuresSizes[field]; element++)
            {
                double learnedValue = (1.0 - learningRate[field]) * learningNeuron[field][element]
                        + learningRate[field] * Math.Min(learningNeuron[field][element], activity[field][element]);
                learningNeuron[field][element] = learnedValue;
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Stamps the received stimulus into the selected neuron cluster weights with the fuzzy ART II operation
        /// </summary>
        /// <param name="field"></param>
        /// <param name="selectedNeuron"></param>
        private void stampNeuronARTII(int field, int selectedNeuron)
        {
            double[][] learningNeuron = neurons[selectedNeuron];
            for (int element = 0; element < featuresSizes[field]; element++)
            {
                double learnedValue = (1.0 - learningRate[field]) * learningNeuron[field][element]
                        + learningRate[field] * activity[field][element];
                learningNeuron[field][element] = learnedValue;
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs the composite learn operation 
        /// </summary>
        /// <param name="selectedNeuron"></param>
        private void learnComposite(int selectedNeuron)
        {
            if (neurons[selectedNeuron][totalFields - 1][0] == 1.0
                    && neurons[selectedNeuron][totalFields - 1][1] == 0.0 && fieldsClass[totalFields - 1] == FieldTypes.REWARD)
                return;

            if (neurons[selectedNeuron][totalFields - 1][0] == 0.0
                    && neurons[selectedNeuron][totalFields - 1][1] == 1.0 && fieldsClass[totalFields - 1] == FieldTypes.REWARD)
                return;

            for (int field = 0; field < totalFields; field++)
            {
                int op = learningOp[field];
                switch (op)
                {
                    case NeuronLearning.ART_I:
                        stampNeuronARTI(field, selectedNeuron);
                        break;
                    case NeuronLearning.ART_II:
                        stampNeuronARTII(field, selectedNeuron);
                        break;
                }
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Resets the last selected neuron action. Helps the action selection when using a reactive model 
        /// </summary>
        public void resetLastNeuronAction()
        {
            double[][] learningNeuron = neurons[lastSelectedNeuron];
            for (int field = 0; field < totalFields; field++)
            {
                if (fieldsClass[field] == FieldTypes.ACTION)
                {
                    for (int element = 0; element < featuresSizes[field]; element++)
                    {
                        double learnedValue = activity[field][element];
                        learningNeuron[field][element] = learnedValue;
                    }
                }
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Overwrites the selected neuron cluster weights with the received stimulus 
        /// </summary>
        /// <param name="selectedNeuron"></param>
        private void overwrite(int selectedNeuron)
        {
            double[][] learningNeuron = neurons[selectedNeuron];
            for (int field = 0; field < totalFields; field++)
            {
                for (int element = 0; element < featuresSizes[field]; element++)
                {
                    double learnedValue = activity[field][element];
                    learningNeuron[field][element] = learnedValue;
                }
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs a readout operation to the current activity using the fuzzy ART I operation 
        /// </summary>
        /// <param name="selectedNeuron"></param>
        private void ARTIReadout(int selectedNeuron)
        {
            double[][] readoutneuron = neurons[selectedNeuron];
            for (int field = 0; field < totalFields; field++)
            {
                for (int element = 0; element < featuresSizes[field]; element++)
                {
                    activity[field][element] = Math.Min(readoutneuron[field][element], activity[field][element]);
                }
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs a readout operation to the current activity using the fuzzy ART II operation 
        /// </summary>
        /// <param name="selectedNeuron"></param>
        private void readout(int selectedNeuron)
        {
            double[][] readoutneuron = neurons[selectedNeuron];
            for (int field = 0; field < totalFields; field++)
            {
                for (int element = 0; element < featuresSizes[field]; element++)
                {
                    activity[field][element] = readoutneuron[field][element];
                }
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Returns the total number of neurons used by the ANN 
        /// </summary>
        /// <returns></returns>
        public int getTotalAmountOfNeurons()
        {
            return neurons.Count;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs the neuron prunning procedure when acting through a reactive model 
        /// </summary>
        public void neuronPrunning()
        {
            int neuronsToPrunne = 0;

            if (neurons.Count >= neuronPrunningThreshold)
            {
                for (int currentNeuron = 0; currentNeuron < neuronsConfidence.Count; currentNeuron++)
                {
                    double confidence = neuronsConfidence[currentNeuron];
                    if (confidence < neuronConfidenceThreshold)
                    {
                        neurons[currentNeuron] = null;
                        neuronsToPrunne++;
                    }
                }
            }

            while (neuronsToPrunne > 0)
            {
                for (int currentNeuron = 0; currentNeuron < neurons.Count - 1; currentNeuron++)
                {
                    if (neurons[currentNeuron] == null)
                    {
                        neurons.RemoveAt(currentNeuron);
                        neuronsConfidence.RemoveAt(currentNeuron);
                        neuronsToPrunne--;
                        break;
                    }
                }
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs the neuron cluster reinforcement operation when using a reactive model 
        /// </summary>
        public void neuronReinforcement()
        {
            double oldConfidence = neuronsConfidence[lastSelectedNeuron];
            double newConfidence = oldConfidence + neuronReinforcementRate * (1.0 - oldConfidence);
            neuronsConfidence[lastSelectedNeuron] = newConfidence;
        }

        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs the neuron cluster erosion operation when using a reactive model 
        /// </summary>
        public void neuronErosion()
        {
            double oldConfidence = neuronsConfidence[lastSelectedNeuron];
            double newConfidence = oldConfidence - neuronErosionRate * oldConfidence;
            neuronsConfidence[lastSelectedNeuron] = newConfidence;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs the neuron cluster decay operation when using a reactive model 
        /// </summary>
        public void neuronDecay()
        {
            for (int currentNeuron = 0; currentNeuron < neuronsConfidence.Count; currentNeuron++)
            {
                double confidence = neuronsConfidence[currentNeuron];
                double newConfidence = confidence - neuronDecayRate * confidence;
                neuronsConfidence[currentNeuron] = newConfidence;
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Set the vigilance parameters to perform or learn 
        /// </summary>
        /// <param name="learning"></param>
        /// <returns></returns>
        private double[] calculateVigilances(bool learning)
        {
            if (learning)
                return (double[])learnVigilances.Clone();
            return (double[])performVigilances.Clone();
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Return the highest value from a double array
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        private static int max(double[] array)
        {
            int result = -1;
            double max = Double.MinValue;
            for (int i = 0; i < array.Length; i++)
            {
                if (array[i] >= max)
                {
                    max = array[i];
                    result = i;
                }
            }
            return result;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Select an action if using an action mask model 
        /// </summary>
        /// <returns></returns>
        public int selectAction()
        {
            prediction(false);

            if (lastSelectedNeuron == neurons.Count - 1) // Predição de neuronio
                return -1;

            int actionField = 0;
            for (int i = 0; i < totalFields; i++)
            {
                if (fieldsClass[i] == FieldTypes.ACTION)
                {
                    actionField = i;
                    break;
                }
            }

            int selectedAct = max(activity[actionField]);
            return selectedAct;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Check if the neuron cluster is resonating with the received stimulus 
        /// </summary>
        /// <param name="vigilances"></param>
        /// <param name="neuron"></param>
        /// <param name="neuronXandWSum"></param>
        /// <returns></returns>
        private bool isResonating(double[] vigilances, double[][] neuron, double[] neuronXandWSum)
        {
            bool resonated = true;
            for (int field = 0; field < totalFields; field++)
            {
                if (activeFields[field])
                {
                    double stateMatchFactor = doMatch(field, neuron[field], neuronXandWSum[field]);

                    if (stateMatchFactor < vigilances[field])
                    {
                        resonated = false;
                        break;
                    }
                }
            }
            return resonated;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Insert, as a copy, the perceived stimulus into the last selected neuron cluster 
        /// </summary>
        public void insert()
        {
            double[][] learningNeuron = neurons[lastSelectedNeuron];
            for (int field = 0; field < totalFields; field++)
            {
                for (int element = 0; element < featuresSizes[field]; element++)
                {
                    double learnedValue = activity[field][element];
                    learningNeuron[field][element] = learnedValue;
                }
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs the prediction operation to select a neuron cluster as a retrieved memory 
        /// </summary>
        /// <param name="learning"></param>
        public void prediction(bool learning)
        {
            // DEBUG PRINTING
            if (debug)
            {
                if (learning)
                    Console.WriteLine("------------ PERFORMING A LEARNING OPERATION ------------------");
                else
                    Console.WriteLine("------------ PERFORMING A PREDICTION OPERATION ----------------");

                Console.WriteLine("Received stimulus: ");
                printNeuron(inputFields);
            }

            // create operational copy of the input stimulus fields
            for (int field = 0; field < totalFields; field++)
                Array.Copy(inputFields[field], activity[field], activity[field].Length);

            calculateActivitySum();

            double[] vigilances = calculateVigilances(learning);

            double[][] wAndxSum = new double[neurons.Count][];
            for (int i = 0; i < wAndxSum.Length; i++)
                wAndxSum[i] = new double[totalFields];

            List<NeuronTemperatureTuple> neuronsTemperature = new List<NeuronTemperatureTuple>();
            for (int currentNeuron = 0; currentNeuron < neurons.Count - 1; currentNeuron++)
            {
                double t = calculateTComposite(neurons[currentNeuron], wAndxSum[currentNeuron]);
                neuronsTemperature.Add(new NeuronTemperatureTuple(currentNeuron, t));
            }
            calculateTComposite(neurons[neurons.Count - 1], wAndxSum[neurons.Count - 1]);
            neuronsTemperature.Add(new NeuronTemperatureTuple(neurons.Count - 1, 0.0));

            neuronsTemperature.Sort((first, second) =>
            {
                if (first != null && second != null)
                    return second.t.CompareTo(first.t);

                if (first == null && second == null)
                    return 0;

                if (first != null)
                    return -1;

                return 1;
            });

            int selectedNeuron = -1;
            bool perfectMissmatch = false;
            for (int i = 0; i < neuronsTemperature.Count; i++)
            {
                int maxT = neuronsTemperature[i].neuronIndex;
                double[][] neuron = neurons[maxT];
                double[] neuronXandWSum = wAndxSum[maxT];
                selectedNeuron = maxT;

                bool resonated = isResonating(vigilances, neuron, neuronXandWSum);

                if (resonated)
                {
                    break;
                }
               /* else
                {
                    bool perfectError = true;
                    for (int field = 0; field < totalFields; field++)
                    {
                        if (!checkPerfectMissmatch(field, neuron) && fieldsClass[field] == FieldTypes.STATE)
                        {
                            perfectError = false;
                            break;
                        }
                    }

                    if (perfectError)
                    {
                        perfectMissmatch = true;
                        break;
                    }
                    else
                    {
                        for (int field = 0; field < totalFields; field++)
                        {
                            if (vigilancesRaising[field])
                            {
                                double stateMatchFactor = doMatch(field, neuron[field], neuronXandWSum[field]);
                                if (stateMatchFactor > vigilances[field])
                                    vigilances[field] = Math.Min(stateMatchFactor + adaptiveVigilanceRate, 1.0);
                            }
                        }
                    }
                }*/
            }

            if (learning)
            {
                if (learningEnabled)
                {
                    if (perfectMissmatch)
                        overwrite(selectedNeuron);
                    else
                        learnComposite(selectedNeuron);

                    if (selectedNeuron == neurons.Count - 1)
                        createNeuron();
                }
            }
            else
            {
                if (fuzzyReadout)
                    ARTIReadout(selectedNeuron);
                else
                    readout(selectedNeuron);
            }

            lastSelectedNeuron = selectedNeuron;

            // DEBUG PRINTING
            if (debug)
            {
                Console.WriteLine("Activity for neuron " + selectedNeuron + ":");
                printNeuron(selectedNeuron);

                Console.WriteLine("Readout for neuron " + selectedNeuron + ":");
                printNeuron(activity);

                Console.WriteLine();
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Performs the prediction operation to select a neuron cluster as a retrieved memory 
        /// </summary>
        /// <param name="learning"></param>
        public void prediction(bool learning, bool functionBehavior)
        {
            // DEBUG PRINTING
            if (debug)
            {
                if (learning)
                    Console.WriteLine("------------ PERFORMING A LEARNING OPERATION ------------------");
                else
                    Console.WriteLine("------------ PERFORMING A PREDICTION OPERATION ----------------");

                Console.WriteLine("Received stimulus: ");
                printNeuron(inputFields);
            }

            // create operational copy of the input stimulus fields
            for (int field = 0; field < totalFields; field++)
                Array.Copy(inputFields[field], activity[field], activity[field].Length);

            calculateActivitySum();

            double[] vigilances = calculateVigilances(learning);

            double[][] wAndxSum = new double[neurons.Count][];
            for (int i = 0; i < wAndxSum.Length; i++)
                wAndxSum[i] = new double[totalFields];

            List<NeuronTemperatureTuple> neuronsTemperature = new List<NeuronTemperatureTuple>();
            for (int currentNeuron = 0; currentNeuron < neurons.Count - 1; currentNeuron++)
            {
                double t = calculateTComposite(neurons[currentNeuron], wAndxSum[currentNeuron]);
                neuronsTemperature.Add(new NeuronTemperatureTuple(currentNeuron, t));
            }
            calculateTComposite(neurons[neurons.Count - 1], wAndxSum[neurons.Count - 1]);
            neuronsTemperature.Add(new NeuronTemperatureTuple(neurons.Count - 1, 0.0));

            neuronsTemperature.Sort((first, second) =>
            {
                if (first != null && second != null)
                    return second.t.CompareTo(first.t);

                if (first == null && second == null)
                    return 0;

                if (first != null)
                    return -1;

                return 1;
            });

            int selectedNeuron = -1;
            bool perfectMissmatch = false;
            for (int i = 0; i < neuronsTemperature.Count; i++)
            {
                int maxT = neuronsTemperature[i].neuronIndex;
                double[][] neuron = neurons[maxT];
                double[] neuronXandWSum = wAndxSum[maxT];
                selectedNeuron = maxT;

                bool resonated = isResonating(vigilances, neuron, neuronXandWSum);

                if (resonated)
                {
                    break;
                }
                else
                {
                    if (functionBehavior)
                    {
                        bool perfectError = true;
                        for (int field = 0; field < totalFields; field++)
                        {
                            if (!checkPerfectMissmatch(field, neuron) && fieldsClass[field] == FieldTypes.STATE)
                            {
                                perfectError = false;
                                break;
                            }
                        }

                        if (perfectError)
                        {
                            perfectMissmatch = true;
                            break;
                        }
                        else
                        {
                            for (int field = 0; field < totalFields; field++)
                            {
                                if (vigilancesRaising[field])
                                {
                                    double stateMatchFactor = doMatch(field, neuron[field], neuronXandWSum[field]);
                                    if (stateMatchFactor > vigilances[field])
                                        vigilances[field] = Math.Min(stateMatchFactor + adaptiveVigilanceRate, 1.0);
                                }
                            }
                        }
                    }
                }
            }

            if (learning)
            {
                if (learningEnabled)
                {
                    if (perfectMissmatch)
                        overwrite(selectedNeuron);
                    else
                        learnComposite(selectedNeuron);

                    if (selectedNeuron == neurons.Count - 1)
                        createNeuron();
                }
            }
            else
            {
                if (fuzzyReadout)
                    ARTIReadout(selectedNeuron);
                else
                    readout(selectedNeuron);
            }

            lastSelectedNeuron = selectedNeuron;

            // DEBUG PRINTING
            if (debug)
            {
                Console.WriteLine("Activity for neuron " + selectedNeuron + ":");
                printNeuron(selectedNeuron);

                Console.WriteLine("Readout for neuron " + selectedNeuron + ":");
                printNeuron(activity);

                Console.WriteLine();
            }
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Returns the last obtained prediction, stored in the activated neuron cluster 
        /// </summary>
        /// <returns></returns>
        public double[][] getLastActivatedPrediction()
        {
            return neurons[lastSelectedNeuron];
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Returns the last obtained prediction index
        /// </summary>
        /// <returns></returns>
        public int getLastActivatedCluster()
        {
            return lastSelectedNeuron;
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Set the activity individually
        /// </summary>
        /// <param name="field"></param>
        /// <param name="externalStimulus"></param>
        public void setInputField(int field, double[] externalStimulus)
        {
            if (externalStimulus == null)
            {
                Console.WriteLine("The external stimulus can not be null.");
                return;
            }

            if (field > activity.Length - 1)
            {
                Console.WriteLine("The requested field does not exist.");
                return;
            }

            double[] operational = new double[externalStimulus.Length];
            Array.Copy(externalStimulus, operational, externalStimulus.Length);

            int actvityLength = inputFields[field].Length;
            int stimulusLength = operational.Length;

            if (actvityLength > stimulusLength)
            {
                Console.WriteLine("The size of the external stimulus is different from the requested field.");
                return;
            }

            Array.Copy(operational, inputFields[field], operational.Length);
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Set the activity individually
        /// </summary>
        /// <param name="field"></param>
        /// <param name="externalStimulus"></param>
        public void setInputField(int field, double[] externalStimulus, int complementCodingType)
        {
            if (externalStimulus == null)
            {
                Console.WriteLine("The external stimulus can not be null.");
                return;
            }

            if (field > activity.Length - 1)
            {
                Console.WriteLine("The requested field does not exist.");
                return;
            }

            double[] operational = new double[externalStimulus.Length];
            Array.Copy(externalStimulus, operational, externalStimulus.Length);

            switch (complementCodingType)
            {
                case ComplementCodingType.NONE:
                    break;
                case ComplementCodingType.MIRRORED:
                    operational = turnComplemented(operational, false);
                    break;
                case ComplementCodingType.PREDICTION:
                    operational = turnComplemented(operational, true);
                    break;
                case ComplementCodingType.DIRECT_ACCESS:
                    operational = generateDirectAccess(operational);
                    break;
            }

            int actvityLength = inputFields[field].Length;
            int stimulusLength = operational.Length;

            if (actvityLength > stimulusLength)
            {
                Console.WriteLine("The size of the external stimulus is different from the requested field.");
                return;
            }

            Array.Copy(operational, inputFields[field], operational.Length);
        }
        //----------------------------------------------------------------------------------------------------
        /// <summary>
        /// Return what was predicted for the selected field
        /// </summary>
        /// <param name="field"></param>
        /// <param name="externalStimulus"></param>
        public double[] readPrediction(int field)
        {
            if (field > activity.Length - 1)
            {
                Console.WriteLine("The requested field does not exist.");
                return null;
            }

            Array.Copy(activity[field], predictionFields[field], activity[field].Length);
            return predictionFields[field];
        }
        /// <summary>
        /// Turns the received stimulus into a mirrored complemented code
        /// </summary>
        private double[] turnComplemented(double[] stimulus, bool prediction)
        {
            int newTotalSize = stimulus.Length * 2;
            double[] newStimulus = new double[newTotalSize];
            Array.Copy(stimulus, newStimulus, stimulus.Length);
            for (int i = stimulus.Length - 1, j = stimulus.Length; i >= 0; i--, j++)
                if (prediction)
                    newStimulus[j] = stimulus[i];
                else
                    newStimulus[j] = 1.0 - stimulus[i];

            return newStimulus;
        }
        /// <summary>
        /// Turns the received stimulus into a mirrored complemented code
        /// </summary>
        private double[] generateDirectAccess(double[] stimulus)
        {
            int newTotalSize = stimulus.Length * 2;
            double[] newStimulus = new double[newTotalSize];
            for (int i = 0; i < newTotalSize; i++)
                newStimulus[i] = 1.0;
            return newStimulus;
        }
        //----------------------------------------------------------------------------------------------------
        public void printNetStructure()
        {
            Console.WriteLine("------------ NETWORK'S STRUCTURE ------------------------------");
            Console.WriteLine("Total number of fields: " + totalFields);

            Console.WriteLine("Field sizes: ");
            for (int i = 0; i < totalFields; i++)
                Console.WriteLine("\t" + featuresSizes[i]);

            Console.WriteLine("Neuron cluster temperature operations: ");
            for (int i = 0; i < totalFields; i++)
            {
                switch (temperatureOp[i])
                {
                    case NeuronActivation.ART_I:
                        Console.WriteLine("\tfuzzyARTI");
                        break;
                    case NeuronActivation.ART_II:
                        Console.WriteLine("\tfuzzyARTII");
                        break;
                    case NeuronActivation.PROXIMITY:
                        Console.WriteLine("\tProximity");
                        break;
                }
            }

            Console.WriteLine("Used operations for learning: ");
            for (int i = 0; i < totalFields; i++)
            {
                switch (learningOp[i])
                {
                    case NeuronLearning.ART_I:
                        Console.WriteLine("\tfuzzyARTI");
                        break;
                    case NeuronLearning.ART_II:
                        Console.WriteLine("\tfuzzyARTII");
                        break;
                }
            }

            Console.WriteLine("Field types: ");
            for (int i = 0; i < totalFields; i++)
            {
                switch (fieldsClass[i])
                {
                    case FieldTypes.STATE:
                        Console.WriteLine("\tSTATE");
                        break;
                    case FieldTypes.ACTION:
                        Console.WriteLine("\tACTION");
                        break;
                    case FieldTypes.REWARD:
                        Console.WriteLine("\tREWARD");
                        break;
                }
            }
            Console.WriteLine("---------------------------------------------------------------");
            Console.WriteLine();
            Console.WriteLine();
        }
        //----------------------------------------------------------------------------------------------------
        public void printNetworkParameters()
        {
            Console.WriteLine("------------ NETWORK'S PARAMETERS -----------------------------");
            Console.WriteLine("Learning rates: ");
            for (int i = 0; i < totalFields; i++)
                Console.WriteLine("\t" + learningRate[i]);

            Console.WriteLine("Learning vigilances: ");
            for (int i = 0; i < totalFields; i++)
                Console.WriteLine("\t" + learnVigilances[i]);

            Console.WriteLine("Performing vigilances: ");
            for (int i = 0; i < totalFields; i++)
                Console.WriteLine("\t" + performVigilances[i]);

            Console.WriteLine("Field gammas: ");
            for (int i = 0; i < totalFields; i++)
                Console.WriteLine("\t" + gammas[i]);

            Console.WriteLine("Field alphas: ");
            for (int i = 0; i < totalFields; i++)
                Console.WriteLine("\t" + alphas[i]);

            Console.WriteLine("Active fields: ");
            for (int i = 0; i < totalFields; i++)
                if (activeFields[i])
                    Console.WriteLine("\tTrue");
                else
                    Console.WriteLine("\tFalse");

            Console.WriteLine("Active adaptive vigilances: ");
            for (int i = 0; i < totalFields; i++)
                if (vigilancesRaising[i])
                    Console.WriteLine("\tTrue");
                else
                    Console.WriteLine("\tFalse");

            Console.WriteLine("Adaptive vigilance rate: " + adaptiveVigilanceRate);
            if (fuzzyReadout)
                Console.WriteLine("Fuzzy readout: Enabled");
            else
                Console.WriteLine("Fuzzy readout: Disabled");
            if (learningEnabled)
                Console.WriteLine("Learning: Enabled");
            else
                Console.WriteLine("learning: Disabled");

            Console.WriteLine("---------------------------------------------------------------");
            Console.WriteLine();
            Console.WriteLine();
        }
        //----------------------------------------------------------------------------------------------------
        public void printNeuron(int neuronToPrint)
        {
            double[][] neuron = neurons[neuronToPrint];

            // Console.WriteLine("Neuron " + neuronToPrint);
            for (int field = 0; field < totalFields; field++)
            {
                Console.Write("\t[");
                for (int i = 0; i < featuresSizes[field]; i++)
                {
                    if (i < featuresSizes[field] - 1)
                        Console.Write(neuron[field][i] + " | ");
                    else
                        Console.Write(neuron[field][i]);
                }
                Console.WriteLine("]");
            }
        }
        //----------------------------------------------------------------------------------------------------
        public void printNeuron(double[][] neuron)
        {
            // Console.WriteLine("Neuron " + neuronToPrint);
            for (int field = 0; field < totalFields; field++)
            {
                Console.Write("\t[");
                for (int i = 0; i < featuresSizes[field]; i++)
                {
                    if (i < featuresSizes[field] - 1)
                        Console.Write(neuron[field][i] + " | ");
                    else
                        Console.Write(neuron[field][i]);
                }
                Console.WriteLine("]");
            }
        }
        //----------------------------------------------------------------------------------------------------
        public void setDebug(bool value)
        {
            this.debug = value;
        }
        //----------------------------------------------------------------------------------------------------
        public List<double[][]> getNeurons()
        {
            return neurons;
        }

        // Structure variables
        private int totalFields;
        private int[] featuresSizes; // ok
        private int[] temperatureOp; // ok
        private int[] learningOp; // ok
        private int[] fieldsClass; // ok

        // Net dynamics variables
        private double[] learningRate; // ok
        private double[] learnVigilances; // ok
        private double[] performVigilances; // ok
        private double[] gammas; // ok
        private double[] alphas; // ok
        private double adaptiveVigilanceRate; // ok

        // Reactive model helper dynamics variables
        private double neuronErosionRate;
        private double neuronReinforcementRate;
        private double neuronDecayRate;
        private double neuronConfidenceThreshold;
        private int neuronPrunningThreshold;

        // Q-Learning model helper dynamics variables
        //private double qDiscountParameter;
        //private double qLearningRate;

        // Control variables
        private bool[] activeFields; // ok
        private bool[] vigilancesRaising; // ok
        private bool fuzzyReadout; // ok
        private bool learningEnabled = true;
        private int lastSelectedNeuron = 0;

        // Operation variables
        private double[][] predictionFields;
        private double[][] inputFields;
        private double[][] activity;
        private double[] activitySum;

        private List<double> neuronsConfidence;
        private List<double[][]> neurons;

        private bool debug;

    }
}
