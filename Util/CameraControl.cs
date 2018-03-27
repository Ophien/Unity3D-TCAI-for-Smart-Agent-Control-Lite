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

public class CameraControl : MonoBehaviour {

    public GameObject back;
    public GameObject front;
    public GameObject roof;
    public GameObject P1;
    public GameObject P2;
    public GameObject P3;

    public void BackCamera()
    {
        back.SetActive(true);
        front.SetActive(false);
        roof.SetActive(false);
        P1.SetActive(false);
        P2.SetActive(false);
        P3.SetActive(false);
    }

    public void FrontCamera()
    {
        front.SetActive(true);
        roof.SetActive(false);
        back.SetActive(false);
        P1.SetActive(false);
        P2.SetActive(false);
        P3.SetActive(false);
    }

    public void RoofCamera()
    {
        roof.SetActive(true);
        front.SetActive(false);
        back.SetActive(false);
        P1.SetActive(false);
        P2.SetActive(false);
        P3.SetActive(false);
    }

    public void CallP1()
    {
        roof.SetActive(false);
        front.SetActive(false);
        back.SetActive(false);
        P1.SetActive(true);
        P2.SetActive(false);
        P3.SetActive(false);
    }

    public void CallP2()
    {
        roof.SetActive(false);
        front.SetActive(false);
        back.SetActive(false);
        P1.SetActive(false);
        P2.SetActive(true);
        P3.SetActive(false);
    }

    public void CallP3()
    {
        roof.SetActive(false);
        front.SetActive(false);
        back.SetActive(false);
        P1.SetActive(false);
        P2.SetActive(false);
        P3.SetActive(true);
    }

    // Use this for initialization
    void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
		
	}
}
