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
using TCAI;

public class Reseter : MonoBehaviour {
    public GameObject startPosition;
    public double resetReward;

    void OnTriggerEnter(Collider col)
    {
        if (col.gameObject.tag == "agent")
        {
            col.gameObject.transform.position = startPosition.transform.position;
            GenericReinforcementAgent qtab = col.gameObject.GetComponent<GenericReinforcementAgent>();
            if (qtab != null)
                qtab.reset(resetReward);

        }
    }

    void OnTriggerStay(Collider col)
    {
        if (col.gameObject.tag == "agent")
        {
            col.gameObject.transform.position = startPosition.transform.position;
            GenericReinforcementAgent qtab = col.gameObject.GetComponent<GenericReinforcementAgent>();
            if (qtab != null)
                qtab.reset(resetReward);
        }
    }
}
