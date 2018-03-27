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
 * Alysson.Ribeiro.Silva@gmail.com entitled as "Q-Learning Agent"
*/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TCAI
{
    public abstract class GenericReinforcementAgent : MonoBehaviour
    {
        // Q learning dynamics
        private float intialEpsilon = 1.0f;
        private float epsilon = 1.0f;
        private float epsilonDecay = 0.001f;
        private float discountParameter = 1.0f;
        private float learningRate = 1.0f;

        // Q learning
        private double Q_old = -1.0;
        private double Q_new = -1.0;
        private double[] prevEnvironment;
        private int selectedAction = 0;
        private double immediateReward = 0.0;

        // Abstract functions for Q-learning
        public abstract List<int> generateAvailableActionsList();
        public abstract void performAction(int action);
        public abstract double calculateReward();
        public abstract double predictQValue(double[] environment, double[] action);
        public abstract double learnStimulus(double[] prevEnv, double[] action, double reward);
        public abstract void reset(double resetReward);
        public abstract void _update();
        public abstract void _start();

        // control
        private int actionCount = 0;

        /// <summary>
        /// Unity engine update method.
        /// </summary>
        private void Update()
        {
            _update();
        }

        /// <summary>
        /// Unity engine start method.
        /// </summary>
        private void Start()
        {
            _start();
        }

        /// <summary>
        /// Created by Alysson Ribeiro da Silva, the SetNumberOfActions set the total number of actions used by the reinforcement algorithm
        /// </summary>
        /// <param name="actionCount"></param>
        public void SetNumberOfActions(int actionCount)
        {
            this.actionCount = actionCount;
        }

        /// <summary>
        /// Created by Alysson Ribeiro da Silva, the generateActionVector generates a boolean action vector to be used by a learning algorithm
        /// </summary>
        /// <param name="selectedAction"></param>
        /// <returns></returns>
        private double[] generateActionVector(int selectedAction)
        {
            double[] ret = new double[actionCount];
            ret[selectedAction] = 1.0;
            return ret;
        }

        /// <summary>
        /// Created by Alysson Ribeiro da Silva, the PhaseOne calls all routines to perform the first part of the Q-learning algorithm (exploration and exploitation)
        /// </summary>
        /// <param name="environment"></param>
        /// <param name="availableActions"></param>
        public void PhaseOne(double[] environment, List<int> availableActions)
        {
            if (Q_old != -1.0)
            {
                PhaseTwo(environment, availableActions);
                Q_old = 1.0;
            }

            // select exploration or exploitation
            float randomFloat = Random.Range(0.0f, 1.0f);
            double[] reward = new double[1];
            double max_q = double.MinValue;
            bool enableRandom = true;

            // Random Action
            if (randomFloat < epsilon && enableRandom)
            {
                // Set variables to move agent
                int randomMoveAct = Random.Range(0, availableActions.Count);
                randomMoveAct = availableActions[randomMoveAct];
                double[] actVec = generateActionVector(randomMoveAct);

                // Calculates the predicted Q-value from your q-table structure
                max_q = predictQValue(environment, actVec);

                // Set selected action
                selectedAction = randomMoveAct;
            }
            // Q-Max action
            else
            {
                for (int i = 0; i < availableActions.Count; i++)
                {
                    // Setup action moving field
                    int field = availableActions[i];
                    double[] actVec = generateActionVector(field);

                    double q_value = predictQValue(environment, actVec);

                    if (q_value > max_q)
                    {
                        max_q = q_value;
                        selectedAction = field;
                    }
                }
            }

            // set the Q-old variable to compute the temporal difference
            Q_old = max_q;

            // move agent to configure its next q-phase
            performAction(selectedAction);

            // store previous state
            prevEnvironment = new double[environment.Length];
            environment.CopyTo(prevEnvironment, 0);
        }

        /// <summary>
        /// Created by Alysson Ribeiro da Silva, the PhaseTwo algorithm performs the second part of the Q-learning algorithm (check and learning)
        /// </summary>
        /// <param name="environment"></param>
        /// <param name="availableActions"></param>
        private void PhaseTwo(double[] environment, List<int> availableActions)
        {
            // Variable to hold the newly expected max Q-value
            Q_new = double.NegativeInfinity;

            // Calculates the immediate reward, used for reinforcement learning
            immediateReward = calculateReward();

            // Check in the expectations, which one is the best and use it to update the Q-learning Q-value of the virtual Q-learning table
            for (int i = 0; i < availableActions.Count; i++)
            {
                // Setup action moving field
                int field = availableActions[i];
                double[] act = generateActionVector(field);

                double q_value = predictQValue(environment, act);

                if (q_value > Q_new)
                {
                    Q_new = q_value;
                }
            }

            // Calculates the Q-learning Temporal Difference error
            double tdErr = learningRate * (immediateReward + (discountParameter * Q_new) - Q_old);

            // Calculates the bounded rule for the reinforcement learning
            //tdErr = tdErr * (1.0 - Q_old);

            // Calculates the newly Q-value to be learned
            double q_to_learn = Q_old + tdErr;

            // Generates the action vector used for learning
            double[] actVec = generateActionVector(selectedAction);

            // Call the learning function to reinforce the observed environment and actions
            learnStimulus(prevEnvironment, actVec, q_to_learn);

            // Updates the epsilon with its decay settings
            epsilon = epsilon - epsilonDecay;
        }

        /// <summary>
        /// Created by Alysson Ribeiro da Silva, the UpdateFinalState function is used to force the Q-learning algorithm to learn stimulus received by the last State
        /// </summary>
        /// <param name="res"></param>
        public void UpdateFinalState(double res)
        {
            // Asign the Q-new variable to perform actions
            Q_new = res;

            // Immediate reward, after performed action
            immediateReward = calculateReward();

            // Tderr formula
            double tdErr = learningRate * (immediateReward + (discountParameter * Q_new) - Q_old);

            // Bounded Q-Learning
            //tdErr = tdErr * (1.0 - Q_old);

            // Q-To be learned
            double q_to_learn = Q_old + tdErr;

            // Generate action vector to be learned
            double[] actVec = generateActionVector(selectedAction);

            // Call the learning function to reinforce the observed environment and actions
            learnStimulus(prevEnvironment, actVec, q_to_learn);
        }
    }
}