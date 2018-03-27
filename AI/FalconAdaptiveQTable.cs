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
    public class FalconAdaptiveQTable
    {
        public FalconAdaptiveQTable()
        {
            neurons = new Dictionary<string, Dictionary<string, double>>();
        }

        string buildString(double[] vec)
        {
            string ret = "";

            for (int i = 0; i < vec.Length; i++)
            {
                int bin_value = (int)vec[i];
                string c_value = bin_value.ToString();
                ret += c_value;
            }

            return ret;
        }

        public double predict(double[] environment, double[] action, double reward)
        {
            string envKey = buildString(environment);
            string actKey = buildString(action);

            if (!neurons.ContainsKey(envKey))
            {
                neurons.Add(envKey, new Dictionary<string, double>());
            }

            Dictionary<string, double> action_reward = neurons[envKey];

            if (!action_reward.ContainsKey(actKey))
            {
                action_reward.Add(actKey, 0.5);
            }

            if (reward >= 0.0)
            {
                action_reward[actKey] = reward;
            }

            return action_reward[actKey];
        }

        public int neuronCount()
        {
            int count = 0;
            foreach (string key in neurons.Keys)
            {
                count += neurons[key].Count;
            }

            return count;
        }

        public Dictionary<string, Dictionary<string, double>> neurons;
    }
}