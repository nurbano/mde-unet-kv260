# AMD Open Hardware 2024
## Monocular Depth estimation for interior scenarios in real time on Kria KV260
From a single RGB image, infer in real-time with a Kria KV260 a depth map using convolutional neural networks.
![Diagram](/diagram.png "Diagram MDE KV260")
### Team number: AOHW-305
Participants:
- Nicolás Urbano Pintos (UTN FRH /CITEDEF)
- Monal Patel Rakeshbhai (UMONS)
Supervisor:
- Carlos Valderrama (UMONS)

Steps:

<img src="steps.png" width="200" height="100">

- Train a U-NET model with NYUDEPTHV2 dataset in pytorch.
- Quantize the model with VITIS AI.
- Compile the quantized model for DPU.
- Evaluate the model in the Kria KV260 

