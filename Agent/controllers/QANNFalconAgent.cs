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
using TCAI;
using UnityEngine.UI;

public class QANNFalconAgent : GenericReinforcementAgent
{
    // Target location threshold
    public double targetThreshold = 0.1;

    // NavMesh agent script
    public UnityEngine.AI.NavMeshAgent navMeshAgent;

    // Main body gameobject
    public GameObject mainBody;

    // Animator controller
    public Animator animator;

    // Set current target
    public Transform currentTargetLocation;

    // Object transformation
    public Transform objectiveTransform;

    // Start position
    public Transform startPosition;

    // Set the previous target location
    private Vector3 prevTargetLoc;

    // Control variables
    private float targetDist;
    private float distToTargert;
    private int walkingHash;

    // Adaptive Neural Network for advanced cognition
    public AdaptiveNeuralNetwork network;

    // The field that will receive the observed stimulus
    int environmentField = 0;
    int actionField = 1;
    int rewardField = 2;

    // Moving variable
    private const int moving = 0;
    private const int stationary = 1;

    // Available directions
    private List<int> availableDirections = new List<int>();

    // Current state
    private int currentState = stationary;

    // Moving references
    public Sensor targetDistance;
    public SensorArray movingFields;
    public SensorArray environment;

    // Objective pos array
    public SensorArray objectiveGrid;

    // Reset control
    private bool resetControl = false;
    private double resetReward = -1.0;
    public double resetTimer = 5.0;
    private double start = 0.0;

    // Control variables
    public Text output_neuron;
    public Text trial;
    public Text epoch;
    public Text trialsuccess;
    public Text accuracy;
    public int trialcounter = 0;
    public int trialSuccessCounter = 0;
    public int epochCounter = 0;
    public int epochSize = 25;
    public double acc = 0.0;

    // Use this for initialization
    public override void _start()
    {
        // Set the total amount of actions used in this agent
        SetNumberOfActions(movingFields.sensors.Count);

        // Set the hash for the walking animation
        walkingHash = Animator.StringToHash("walking");

        // Initialize the adaptive neural network structure with a Fusion Architecture for Learning COgnition and Navigation
        // ---------------------------------------------------------------------
        // ------------ Creating a configuration file --------------------------
        // ---------------------------------------------------------------------
        NetDescription description = new NetDescription();

        // ---------------------------------------------------------------------
        // ------------ Configuring an one field Adaptive Neural Network -------
        // ---------------------------------------------------------------------
        description.activeFields = new bool[] { true, true, false }; // defines which fields are active to perform the categorization
        description.adaptiveVigilanceRaising = new bool[] { true, false, false }; // defines which fields will be affected by the adaptive vigilance
        description.fuzzyReadout = false; // defines if the prediction will be performed by the fuzzy ARTI readout

        description.featuresSizes = new int[] { environment.sensors.Count * 2, movingFields.sensors.Count, 2 }; // defines the fields configuration, their size and quantity
        description.alphas = new double[] { 0.1, 0.1, 0.1 }; // defines each field alpha
        description.gammas = new double[] { 0.5, 0.5, 0.0 }; // defines each field gamma
        description.learningRate = new double[] { 1.0, 1.0, 1.0 }; // defines the learning rate for each field
        description.learningVigilances = new double[] { 0.9, 1.0, 0.0 }; // defines the vigilance used for learning for each field
        description.performingVigilances = new double[] { 0.0, 0.0, 0.0 }; // defines the vigilance used to perform predictions for each field
        description.adaptiveVigilanceRate = 0.001; // defines how much the vigilance will be added when predicting

        description.fieldsClass = new int[] { FieldTypes.STATE, FieldTypes.ACTION, FieldTypes.REWARD }; // defines each field type
        description.learningOp = new int[] { NeuronLearning.ART_II, NeuronLearning.ART_II, NeuronLearning.ART_II }; // defines which composite operation will be used by the network when learning
        description.temperatureOp = new int[] { NeuronActivation.ART_I, NeuronActivation.ART_I, NeuronActivation.ART_II }; // defines which composite operation will be used by the network when performing

        // ---------------------------------------------------------------------
        // ---------- Creating the Adaptive Neural Network ---------------------
        // ---------------------------------------------------------------------
        network = new AdaptiveNeuralNetwork(description); // creates the network
    }

    // Update is called once per frame
    public override void _update()
    {
        MoveUntilReachingTarget();
        Logic();

        if (resetControl)
        {
            GetAvailableMovingPositions();

            // Reseting bot position
            mainBody.transform.position = startPosition.transform.position;
            mainBody.transform.rotation = startPosition.transform.rotation;
            movingFields.transform.position = startPosition.transform.position;
            move(0);

            // Reseting objective position
            int randomGrid = Random.Range(0, objectiveGrid.sensorArray.Length);
            Sensor objPos = objectiveGrid.sensors[randomGrid];
            objectiveTransform.position = objPos.transform.position;

            // Reseting start position
            int randomStart = Random.Range(0, objectiveGrid.sensorArray.Length);
            objPos = objectiveGrid.sensors[randomGrid];
            startPosition.position = objPos.transform.position;

            randomMove();

            resetControl = false;

            if (resetReward == 1.0 || resetReward == 0.0)
            {
                UpdateFinalState(resetReward);
            }

            if (resetReward == 1.0)
            {
                // Success
                trialSuccessCounter++;
            }

            resetReward = -1.0;

            epochCounter++;

            acc = (double)trialSuccessCounter / (double)epochCounter;

            if (epochCounter > epochSize)
            {
                trialcounter++;
                epochCounter = 0;
                trialSuccessCounter = 0;
            }
        }

        start += Time.deltaTime;

        // Update neuron count
        output_neuron.text = network.getTotalAmountOfNeurons().ToString();
        trial.text = trialcounter.ToString();
        epoch.text = epochCounter.ToString();
        trialsuccess.text = trialSuccessCounter.ToString();
        string accS = string.Format("{0:N2}", acc);
        accuracy.text = accS;

        if (start > resetTimer)
            reset(-1.0);
    }

    /// <summary>
    /// Move function. Move the agent until reaching a target location. Is called several times inside the logic function.
    /// </summary>
    void MoveUntilReachingTarget()
    {
        targetDist = Vector3.Distance(prevTargetLoc, currentTargetLocation.position);
        if (targetDist != 0.0)
        {
            navMeshAgent.destination = currentTargetLocation.position;
            navMeshAgent.isStopped = false;
            currentState = moving;
        }

        if (currentState != moving)
            animator.SetBool(walkingHash, false);
        else
            animator.SetBool(walkingHash, true);

        distToTargert = Vector3.Distance(mainBody.transform.position, currentTargetLocation.position);
        if (distToTargert < targetThreshold)
        {
            navMeshAgent.isStopped = true;
            currentState = stationary;
        }
        prevTargetLoc = currentTargetLocation.position;
    }

    /// <summary>
    /// Gets a moving field for movement. Gets the transform to set as a target location to perform movement.
    /// </summary>
    /// <param name="direction"></param>
    /// <returns></returns>
    Transform GetPositionTransform(int direction)
    {
        Transform ret = movingFields.sensors[direction].transform;

        return ret;
    }

    /// <summary>
    /// Move function acoording to reference. Set target location to a random one.
    /// </summary>
    void randomMove()
    {
        int randomMoveAct = Random.Range(0, availableDirections.Count);
        randomMoveAct = availableDirections[randomMoveAct];
        currentTargetLocation.transform.position = GetPositionTransform(randomMoveAct).position;
    }

    /// <summary>
    /// Move function acoording to reference. Simple set the target moving location for the agent
    /// </summary>
    /// <param name="movingField"></param>
    void move(int movingField)
    {
        currentTargetLocation.transform.position = GetPositionTransform(movingField).position;
    }

    /// <summary>
    /// Logic update
    /// </summary>
    void Logic()
    {
        // Check available moving positions for the reinforcement agent
        GetAvailableMovingPositions();

        // State machine for the nav mesh agent
        switch (currentState)
        {
            // If stationary, select a destination through the reinforcement learning technique and perfom an action
            case stationary:
                if (availableDirections.Count == 0)
                    move(5);
                else
                {
                    PhaseOne(environment.sensorArray, availableDirections);
                }
                break;
            // If moving, do nothing until stop moving ( reaching destination )
            case moving:
                break;
        }
    }

    /// <summary>
    /// Verifies available moving position
    /// </summary>
    void GetAvailableMovingPositions()
    {
        // Clear list to see which actions are available to be used
        availableDirections.Clear();

        // Fill the array with the valid moving positions IDs
        for (int i = 0; i < movingFields.sensorArray.Length; i++)
        {
            if (movingFields.sensorArray[i] == 1.0)
                availableDirections.Add(i);
        }
    }

    // --------------------- TCAI generic reinforcement agent override ---------------------
    /// <summary>
    /// Reset function can be used in special cases where reseting the agent's variables seems to be usefull to promote the simulation logic
    /// </summary>
    /// <param name="resetReward"></param>
    public override void reset(double resetReward)
    {
        start = 0.0;
        this.resetReward = resetReward;
        resetControl = true;
    }

    /// <summary>
    /// Generates all available actions for an observed environment
    /// </summary>
    /// <returns></returns>
    public override List<int> generateAvailableActionsList()
    {
        List<int> availableActions = new List<int>();
        return availableActions;
    }

    /// <summary>
    /// Tells the agent to performe a selected action selected by the reinforcement method
    /// </summary>
    /// <param name="action"></param>
    public override void performAction(int action)
    {
        move(action);
    }

    /// <summary>
    /// Tells how the immediate reward is calculated for the reinforcement learning
    /// </summary>
    /// <returns></returns>
    public override double calculateReward()
    {
        return 1.0 - targetDistance.sensorValue;
    }

    /// <summary>
    /// Tells which Q-value will be used when predicting for the temporal learning method
    /// </summary>
    /// <param name="environment"></param>
    /// <param name="action"></param>
    /// <returns></returns>
    public override double predictQValue(double[] environment, double[] action)
    {
        double[] reward = new double[1];

        // Inserting the observation into the network's activity vectors
        network.setInputField(environmentField, environment, ComplementCodingType.MIRRORED);
        network.setInputField(actionField, action);
        network.setInputField(rewardField, reward, ComplementCodingType.DIRECT_ACCESS);

        // Search q-value
        network.prediction(false);

        // Reading prediction with a complement coding scheme
        double[] predictedReward = network.readPrediction(rewardField);
        if (predictedReward[0] == 1.0 && predictedReward[1] == 1.0)
            return predictedReward[0] / (predictedReward[0] + predictedReward[1]);

        return predictedReward[0];
    }

    /// <summary>
    /// Learns the calculated temporal error
    /// </summary>
    /// <param name="environment"></param>
    /// <param name="action"></param>
    /// <param name="reward"></param>
    /// <returns></returns>
    public override double learnStimulus(double[] prevEnv, double[] action, double reward)
    {
        double[] rwd = new double[1];
        rwd[0] = reward;

        // Inserting the observation into the network's activity vectors
        network.setInputField(environmentField, prevEnv, ComplementCodingType.MIRRORED);
        network.setInputField(actionField, action);
        network.setInputField(rewardField, rwd, ComplementCodingType.MIRRORED);

        // Search q-value
        network.prediction(true);

        // Reading prediction with a complement coding scheme
        double[] predictedReward = network.readPrediction(rewardField);
        if (predictedReward[0] == 1.0 && predictedReward[1] == 1.0)
            return predictedReward[0] / (predictedReward[0] + predictedReward[1]);

        return predictedReward[0];
    }
}
