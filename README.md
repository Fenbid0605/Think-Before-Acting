# Think Before Acting
A Multimodal Service Robot Based on Visual-Tactile-Auditory Integration

Developed by [@Fenbid0605](https://github.com/Fenbid0605) and [@0xJacky](https://github.com/0xJacky)

English | [简体中文](./README-zh_CN.md)

## Project Overview
### Introduction
Despite the continuous development of home service robot technology in recent years, most systems still rely on preset scripts or fixed instructions for execution, lacking genuine understanding of human language, intentions, and preferences. In other words, existing robots are more like "mechanical executors" rather than "intelligent assistants" capable of autonomously adapting to complex home scenarios. For robots to truly integrate into human life, their core capabilities should not be limited to perception and action execution, but should also include semantic understanding of human language, autonomous thinking about tasks, and continuous learning from user behavior. This research proposes a multimodal hierarchical agent system aimed at building home robots with understanding, thinking, and learning capabilities, as shown in Figure 1. The system integrates speech, vision, and tactile sensory signals, based on the Interpreter–Commander–Executor three-layer architecture, converting natural language instructions into specific executable action sequences, and achieving low-latency local inference through edge devices like Jetson Orin Nano. Unlike traditional robots, our system can not only complete simple grasping tasks but also dynamically decompose multi-step operations and achieve more personalized home services based on user preference modeling.

![Figure 1. Task Example](./resources/pic1.png)
<center>Figure 1. Task Example</center>

### System Framework
As shown in Figure 2, this system adopts the Interpreter–Commander–Executor three-layer modular architecture to achieve a complete closed-loop process from natural language instructions to low-level execution control.
Interpreter: Responsible for transcribing user speech into text and performing structured semantic parsing through large language models, extracting task objectives and key parameters.
Commander: Further calls large language models to complete action sequence generation and task dependency structure construction, forming a clear execution graph (DAG).
Executor: Deployed on the edge computing platform Jetson Orin Nano, maps high-level tasks to specific control instructions for robotic arms, grippers, and multimodal sensors through lightweight MCP protocol, while implementing low-latency real-time feedback and dynamic adjustment based on ROS 2, building a closed-loop system of "top-down task instruction chain" and "bottom-up perception feedback flow".
This hierarchical architecture provides robots with good scalability, real-time performance, and task generality during task response, offering system-level support for multimodal interaction and personalized services.

![Figure 2. Overall System Framework](./resources/pic2.png)
<center>Figure 2. Overall System Framework</center>

### Multimodal Hardware Integration
Auditory: iFlytek Speech Module

Visual: Intel Realsense D437 Camera

Tactile: Gelsight Mini Tactile Sensor

Edge Computing: Jetson Orin Nano

## System Functions and Highlights
### Single Task Execution: Grasping a Cup
When a user issues the voice command "Please grab a cup", the system first parses it into structured semantic intent through the Interpreter layer: "Action Type = Grasp; Target Object = Cup". Based on this intent, the Commander layer generates a three-step action sequence: Locate Cup → Plan Grasping Pose → Control Gripper Closure. The Executor layer then uses the Intel RealSense D437 camera to obtain RGB-D images, combines real-time feedback from the Gelsight Mini tactile sensor to fine-tune grasping force, and coordinates with a seven-degree-of-freedom robotic arm and Robotiq gripper to complete the operation. This process completes full local inference on the edge computing device Jetson Orin Nano, with overall response latency controlled at the millisecond level, ensuring real-time performance and stability of the grasping process.

### Multi-task Decomposition Execution: Organizing All Cups on the Table
For the compound command "organize all the cups on the table", the system needs to support automatic task decomposition and dynamic scheduling. The Commander layer, based on the output of the first large language model call, decomposes natural language intent into basic action units such as multi-object detection → sequential grasping → sequential placement, and constructs a directed acyclic graph (DAG) containing dependency relationships. During execution, the system reuses the single-cup grasping control process and dynamically records completed and pending object lists through the state machine module, thus precisely calling corresponding actions in each cycle, ensuring efficient overall process, no omissions, and avoiding potential robotic arm path conflicts. See Figure 3 for specific process examples.

![Figure 3. Multi-task Scenario Example: Decomposition and Execution of Table Cleaning Task](./resources/pic3.png)
<center>Figure 3. Multi-task Scenario Example: Decomposition and Execution of Table Cleaning Task</center>

### Personalized Preference Modeling: Pick Up My Mother's Cup
Facing instructions with subjective preferences like "pick up my mother's cup", the system needs to achieve semantic-perception alignment and individualized reasoning based on semantic understanding. First, in the entity disambiguation module, the system maps "mother" to a specific user identity, then combines color, shape, and other attributes extracted from visual recognition results to locate the object uniquely identified as cup_0032. Meanwhile, the system queries the user preference knowledge graph built on Neo4j graph database to confirm the association relationship between the cup and current family members, and adjusts target priority accordingly. Finally, the Executor layer completes the grasping operation of the specified object. The entire process introduces the TensorFlow Federated framework for local incremental learning on Jetson Orin Nano, uploading only encrypted model update gradients, achieving continuous optimization of personalized strategies while considering user privacy protection. See Figure 4 for specific process examples.

![Figure 4. Preference-Driven Scenario Example: Identifying and Grasping Mother's Cup](./resources/pic4.png)
<center>Figure 4. Preference-Driven Scenario Example: Identifying and Grasping Mother's Cup</center>

## MCP Server-Client Design
The system uses lightweight MCP as the backbone, running both server and client on Jetson Orin Nano. The server is responsible for unified device registration, receiving task intentions from the Interpreter, and routing parsed action flows to the client; the client, as a ROS2 node, calls drivers for robotic arms, Robotiq Gripper, Intel Realsense D437 camera, and Gelsight Mini tactile sensor, executes in real-time and returns sensor data.

## Agentic AI Platform Implementation
Within the platform, the system is divided into three Agents:
1. Interpreter Agent parses voice text and outputs structured intent;
2. Commander Agent combines multimodal perception and user preferences to formulate step plans;
3. Executor Agent drives robotic arm execution through device toolkits.

The three are connected through the Action Graph provided by the platform, with visual orchestration and complete logs; user preferences are written to Memory, enabling the robot to self-learn through interaction.

## UI Design and Optimization
1. Visual Action Graph displays task decomposition and execution progress, supporting node-level backtracking
2. Semantic query panel allows retrieval and manual correction of "preference-item" mappings
3. Homepage displays task queue and execution status in real-time, providing one-click termination or reordering operations, ensuring controllability and safety.

## Key Technologies and Innovations
1. Multimodal Input Fusion
The system combines three types of perceptual information: using Realsense D437 camera to obtain visual data, identifying object colors, positions, and shapes; using tactile sensors to obtain contact surface information and grasping feedback; using far-field microphone ring array ROS six-microphone voice module to obtain user voice input. The three perceptions work together, enabling the system to have more comprehensive environmental understanding and action adjustment capabilities when facing different scenarios.
2. Support for Understanding and Responding to Personalized Semantic Preferences
The project focuses on processing capabilities for "personalized household instructions". The system can recognize ambiguous expressions and subjective preferences in instructions (such as "get my most commonly used cup", "the one my mother likes", "put the book back in its original position", "get the spoon on the left", "use a new bowl, not the old one"), and make reasonable judgments combined with scene perception, demonstrating flexible mapping capabilities from semantics to actions.
3. MCP Protocol Bridging Semantics-Control
Using MCP protocol to encapsulate Interpreter→Commander→Executor signals into unified messages: Intent → Control → Feedback, achieving fast closed-loop within Jetson, with upper layers only caring about intent and lower-layer devices executing by fields, reducing interface changes and making maintenance easier.
4. Edge Deployment Based on Nvidia Jetson Orin Nano
All modules are deployed and run on the Jetson platform, covering semantic processing, perception recognition, and motion control, supporting localized processing and closed-loop feedback, reducing communication latency, improving system stability, suitable for deployment in actual home scenarios.
5. NVIDIA SDK Usage
JetPack 6 integrates CUDA/cuDNN/TensorRT, implementing LLM quantized inference and object detection on Jetson Orin Nano. Core models are exported to ONNX and then unified inference through ONNX Runtime, facilitating seamless migration between development machines and Jetson. VS Code Remote-SSH supports ROS2 remote debugging; optional Azure IoT Hub for OTA updates and log aggregation, automatically falling back to local operation in offline scenarios. Power BI embedded dashboard visualizes MCP logs and grasping metrics, convenient for operations and demonstration.

## Contributions
1. Validated three-layer multimodal agent + preference injection architecture in household robot scenarios, forming end-to-end closed-loop.
2. Built unified MCP message channels, reducing interface fragmentation caused by hardware heterogeneity.
3. Implemented complete edge deployment solutions, providing deployable templates for homes without network access.

## Future Prospects
In the future, we plan to further expand system capabilities from two dimensions: on one hand, at the application scenario level, we hope to migrate this system from household service scenarios to broader practical applications, including assisted object retrieval and interaction in elderly care, intelligent item delivery in hospitals, and food delivery and automatic meal preparation in semi-structured environments. This cross-scenario migration capability will rely on the system's universal perception-understanding-execution closed-loop architecture and continuous learning capabilities, providing service robots with higher adaptability and generality. On the other hand, at the interaction mechanism level, we will further enhance task processing complexity, introduce multi-turn dialogue mechanisms to support continuous, contextually relevant human-machine interaction, enabling the system to understand longer-span semantic chains and dynamically adjust task execution strategies during conversations. Additionally, we plan to optimize resource scheduling and quantization compression strategies for large model inference and edge incremental learning to improve real-time inference efficiency of Jetson Orin Nano in multi-task, multimodal scenarios. Through the above expansions, we are committed to promoting the evolution of home service robots from "responsive tools" to "understanding assistants", moving toward a future of truly "autonomous consciousness" intelligent agents.
