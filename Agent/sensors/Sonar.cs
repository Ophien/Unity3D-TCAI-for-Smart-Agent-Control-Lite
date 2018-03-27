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

public class Sonar : Sensor
{
    // Sonar behavior
    public float sonarDistance = 50.0f;

    // Layer collision mask
    public LayerMask activeLayers;

    // Update is called once per frame
    void Update()
    {
        Vector3 forward = transform.forward;

        Ray front_ray = new Ray(transform.position, forward);
        RaycastHit front_hit;

        if (Physics.Raycast(front_ray, out front_hit, sonarDistance, activeLayers))
        {
            // Control variables
            Vector3 draw = front_hit.point;
            draw.y += 1;

            // Draw lines
            Debug.DrawLine(transform.position, front_hit.point, Color.red);

            // Draw hits
            Debug.DrawLine(front_hit.point, draw, Color.green);

            // Filter collision
            /* if (front_hit.collider.gameObject.tag == "wall")
             {
                 sensorValue = 0.0;
             }

             if(front_hit.collider.gameObject.tag == "mine")
             {
                 sensorValue = 0.5;
             }

             if (front_hit.collider.gameObject.tag == "end")
             {
                 sensorValue = 1.0;
             }*/

            // Compute target distance
            //sensorValue = 1.0 - (1.0 / (1.0 + Vector3.Distance(front_hit.point, transform.position)));

            double dist = Vector3.Distance(front_hit.point, transform.position);
            if (dist > 31.0)
                dist = 31.0;
            sensorValue = dist / 31.0;

            /*if (front_hit.collider.gameObject.tag == "mine")
            {
                sensorValue = 1.0;
            }*/

            //if (sensorValue > 1.0)
            //    sensorValue = 1.0;
            // if (sensorValue < 0.0)
            //    sensorValue = 0.0;
        }
        else
        {
            sensorValue = 0.0;
        }
    }
}
