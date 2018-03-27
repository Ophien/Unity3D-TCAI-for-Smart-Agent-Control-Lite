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

public class ObjectiveDirection : Sensor {
    public GameObject objective;
	
	// Update is called once per frame
	void Update () {
        Vector3 objToSensor = objective.transform.position - transform.position;

        sensorValue = Vector3.Angle(transform.forward, objToSensor);
        if (sensorValue > 180)
            sensorValue = 180;
        if (sensorValue < 0.0)
            sensorValue = 0.0;
        sensorValue /= 180.0;

        Debug.DrawLine(transform.position, transform.position + transform.forward * 2, Color.blue);
        Debug.DrawLine(objective.transform.position, objective.transform.position + objective.transform.forward * 2, Color.cyan);
        Debug.DrawLine(transform.position, objective.transform.position, Color.magenta);
        

        /*if (sensorValue > 1.0)
            sensorValue = 1.0;
        if (sensorValue < 0.0)
            sensorValue = 0.0;*/
    }
}
