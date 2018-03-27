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

public class DirectionSensor : Sensor {

    public GameObject target;

    public const int left = 0;
    public const int up = 1;
    public const int down = 2;
    public const int right = 3;
    public const int sameX_up = 4;
    public const int sameX_down = 5;
    public const int sameY_left = 6;
    public const int sameY_right = 7;

    private int interval = 0;
    private int sabeLayer = 2;
    public int direction;

    // Update is called once per frame
    void Update () {
        Vector3 targetPos = target.transform.position;
        Vector3 myPos = transform.position;

        sensorValue = 0.0;

        switch (direction)
        {
            case left:
                if (myPos.x < targetPos.x - interval && (myPos.z < targetPos.z - interval || myPos.z > targetPos.z + interval))
                    sensorValue = 1.0;
                break;
            case right:
                if (myPos.x > targetPos.x + interval && (myPos.z < targetPos.z - interval || myPos.z > targetPos.z + interval))
                    sensorValue = 1.0;
                break;
            case up:
                if (myPos.z > targetPos.z + interval && (myPos.x < targetPos.x - interval || myPos.x > targetPos.x + interval))
                    sensorValue = 1.0;
                break;
            case down:
                if (myPos.z < targetPos.z - interval && (myPos.x < targetPos.x - interval || myPos.x > targetPos.x + interval))
                    sensorValue = 1.0;
                break;
            case sameY_left:
                if (myPos.z < targetPos.z + sabeLayer && myPos.z > targetPos.z - sabeLayer && myPos.x < targetPos.x - sabeLayer)
                    sensorValue = 1.0;
                break;
            case sameY_right:
                if (myPos.z < targetPos.z + sabeLayer && myPos.z > targetPos.z - sabeLayer && myPos.x > targetPos.x + sabeLayer)
                    sensorValue = 1.0;
                break;
            case sameX_up:
                if (myPos.x < targetPos.x + sabeLayer && myPos.x > targetPos.x - sabeLayer && myPos.z > targetPos.z + sabeLayer)
                    sensorValue = 1.0;
                break;
            case sameX_down:
                if (myPos.x < targetPos.x + sabeLayer && myPos.x > targetPos.x - sabeLayer && myPos.z < targetPos.z - sabeLayer)
                    sensorValue = 1.0;
                break;
        }

       // sensorValue = 1.0 - sensorValue;

    }
}
