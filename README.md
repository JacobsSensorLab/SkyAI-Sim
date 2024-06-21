# SkyAI Sim: An Open-Source Simulation of UAV Aerial Imaging from Satellite Data

Capturing real-world aerial images is challenging due to limited availability and conditions that make it nearly impossible to access all desired images from any location. The complexity increases when multiple locations are involved. Traditional solutions, such as flying a UAV (Unmanned Aerial Vehicle) to take pictures or using existing research databases, have significant limitations. SkyAI Sim offers a compelling alternative by simulating a UAV to capture aerial images with real-world visible-band specifications. This open-source tool allows users to specify the top-left and bottom-right coordinates of any region on a map. Without the need to physically fly a drone, the virtual UAV performs a raster search to capture satellite images using the Google Maps API. Users can define parameters such as flight altitude in meters, aspect ratio and diagonal field of view of the camera, and the overlap between consecutive images. SkyAI Sim's capabilities range from capturing a few low-altitude images for basic applications to generating extensive datasets of entire cities for complex tasks like deep learning. This versatility makes SkyAI a valuable tool for various applications in environmental monitoring, construction, and city management. The open-source nature of the tool also allows for extending the raster search to any other desired path. It is important to note that while the search can theoretically be conducted worldwide, the most accurate results are achieved if users stay within the same or nearby UTM (Universal Transverse Mercator) zone. Therefore, specifying the zone for the program is recommended. Using SkyAI Sim, a few examples of Memphis, TN are also provided.


## Google Colab Repository for local edits:

https://colab.research.google.com/drive/1Huaq96ssyPMy7Xx1IVcUQaoPZu0Idfhk?usp=sharing

<details>
  <summary>How to run colab on local server machine:</summary>

1. SSH to the remote directory and forward the port such as:

```
ssh -L localhost:18888:localhost:8888 [username]@[hostname or IP address]
```

2. Run the following in the remote terminal:
```
jupyter notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=8888 \
    --NotebookApp.port_retries=0
```
2. The previous step will give you two URLs in result. Copy either. For example:
```
http://localhost:8888/?token=0f96a96950ca8aa79c52fb1fa5758e648b5052cd91417dd8
```
or
```
http://127.0.0.1:8888/?token=0f96a96950ca8aa79c52fb1fa5758e648b5052cd91417dd8
```
2. On the bar above select the arrow next to the connect button and choose "connect to a local runtime".
2. A popup window will be shown, paste the copied URL in the input section and change 8888 in it to 18888. For the above example will be:
```
http://localhost:18888/?token=0f96a96950ca8aa79c52fb1fa5758e648b5052cd91417dd8
```
or
```
http://127.0.0.1:18888/?token=0f96a96950ca8aa79c52fb1fa5758e648b5052cd91417dd8
```
3. Press "Connect" and voila.
</details>


## Initiate and activate the environment:

    conda create -n "skyai-sim" python=3.8.5
    conda activate skyai-sim
    cd /path/to/skyai-sim

    pip install -r requirements.txt


## Train with 100 epochs:

    python -m src.main --data_dir /path/to/dataset

for more configuration parameters checkout src/utils/config.py or type:

    python -m src.main --help


## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
