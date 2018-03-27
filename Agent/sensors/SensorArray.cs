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
 * Alysson.Ribeiro.Silva@gmail.com entitled as "Agent"
*/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SensorArray : MonoBehaviour {
    public List<Sensor> sensors;

   // public List<double> featureScaleMin;
   // public List<double> featureScaleMax;

    public double[] sensorArray;

	// Use this for initialization
	void Start () {
        sensorArray = new double[sensors.Count];
	}
	
	// Update is called once per frame
	void Update () {
        int i = 0;
		foreach(Sensor sensor in sensors)
        {
            // if (sensor.sensorValue > featureScaleMax[i])
            //     sensor.sensorValue = featureScaleMax[i];
            // if (sensor.sensorValue < featureScaleMin[i])
            //     sensor.sensorValue = featureScaleMin[i];

            sensorArray[i] = sensor.sensorValue;// (sensor.sensorValue - featureScaleMin[i]) / (featureScaleMax[i] - featureScaleMin[i]);
            i++;
        }
	}
}
