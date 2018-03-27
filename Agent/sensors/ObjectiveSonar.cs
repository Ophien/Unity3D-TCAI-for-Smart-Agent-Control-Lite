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

public class ObjectiveSonar : Sensor {
    public GameObject destination;
	
	// Update is called once per frame
	void Update () {
        Debug.DrawLine(transform.position, destination.transform.position);
        double dist = Vector3.Distance(destination.transform.position, transform.position);
        if (dist > 31.0)
            dist = 31.0;
        sensorValue = dist / 31.0;
    }
}
