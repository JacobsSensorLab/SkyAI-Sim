# SkyAI Sim: An Open-Source Simulation of UAV Aerial Imaging from Satellite Data

Capturing real-world aerial images is challenging due to limited availability and conditions that make it nearly impossible to access all desired images from any location. The complexity increases when multiple locations are involved. Traditional solutions, such as flying a UAV (Unmanned Aerial Vehicle) to take pictures or using existing research databases, have significant limitations. SkyAI Sim offers a compelling alternative by simulating a UAV to capture aerial images with real-world visible-band specifications. This open-source tool allows users to specify the top-left and bottom-right coordinates of any region on a map. Without the need to physically fly a drone, the virtual UAV performs a raster search to capture satellite images using the Google Maps API. Users can define parameters such as flight altitude in meters, aspect ratio and diagonal field of view of the camera, and the overlap between consecutive images. SkyAI Sim's capabilities range from capturing a few low-altitude images for basic applications to generating extensive datasets of entire cities for complex tasks like deep learning. This versatility makes SkyAI a valuable tool for various applications in environmental monitoring, construction, and city management. The open-source nature of the tool also allows for extending the raster search to any other desired path. It is important to note that while the search can theoretically be conducted worldwide, the most accurate results are achieved if users stay within the same or nearby UTM (Universal Transverse Mercator) zone. Therefore, specifying the zone for the program is recommended. Using SkyAI Sim, a few examples of Memphis, TN are also provided.


## Google Colab Repository for local edits:

https://colab.research.google.com/drive/1Huaq96ssyPMy7Xx1IVcUQaoPZu0Idfhk?usp=sharing

<details>
  <summary>How to run colab on local server machine:</summary>

1. SSH to the remote directory and forward the port such as:

```
ssh -L localhost:8888:localhost:8888 [username]@[hostname or IP address]
```

2. Run the following in the remote terminal:
```
jupyter notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=8888 \
    --NotebookApp.port_retries=0
```
3. The previous step will give you two URLs in result. Copy either. For example:
```
http://localhost:8888/?token=0f96a96950ca8aa79c52fb1fa5758e648b5052cd91417dd8
```
or
```
http://127.0.0.1:8888/?token=0f96a96950ca8aa79c52fb1fa5758e648b5052cd91417dd8
```
4. On the bar above select the arrow next to the connect button and choose "connect to a local runtime".
5. A popup window will be shown, paste the copied URL in the input section.
6. Press "Connect" and voila.
7. If you are using a conda environment for your packages, you might need the following steps. On the remote server, install ipykernel:
```
conda install ipykernel
```
1. Then, register the Conda environment as a Jupyter/Colab kernel (Replace <environment_name> with the name of your Conda environment):
```
!python -m ipykernel install --user --name=<environment_name>
```
1. After installing and registering the kernel, you can switch to it from within your Colab notebook interface by selecting it from the kernel dropdown menu (click on "Runtime" > "Change runtime type" > select your Conda environment).
2.  Always restart the Colab runtime after setting up a custom kernel or installing packages to ensure the changes take effect. Click on "Runtime" in the menu and select "Restart runtime...".
</details>


## Initiate and activate the environment:

    conda create -n "skyai-sim" python=3.11
    conda activate skyai-sim
    cd /path/to/skyai-sim

### to enable GPU access (update with your desired version)

check compatibility here: https://www.tensorflow.org/install/source#gpu

    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1 # Not necessary for this project
    pip install tensorflow==2.11

### Install other dependencies:

    pip install -r requirements.txt

## Create a GoogleMap API key:
**(you can skip this if you are only willing to use the data available in dataset folder)**
Check how you can create one here: https://developers.google.com/maps/documentation/javascript/get-api-key


## Access and use the data:

Check notebooks folder.

## Download the Data:

### A single coordinate

To download a sample single image you can run:

    python -m src.download_single

To specify more features, you can do either:

    python -m src.download_single --coords /path/to/file --aspect_ratio <X> <Y> --fov <degrees> --data_dir /path/to/dataset

The file should have one line including the following data:

    <latitude> <longitude> <AGL(f)>

Check dataset/sample_coords.txt as an example.

Or:

    python -m src.download_single --coords "<Latitude>_<Longitude>_<AGL(feet)>"

for more configuration parameters checkout src/utils/config.py or type:

    python -m src.main --help

**For example:**

    python -m src.download_single --coords "dataset/sample_coords.txt" --aspect_ratio 4 3 --fov 78.8 --data_dir dataset/

Note: the aspect ratio and fov are from DJI Mavic and are set to the above values by default.
or

    python -m src.download_single --coords "35.22_-90.07_400"

### A list of coordinates

    python -m src.download_from_list --coords /path/to/file

### Raster Mission
    python -m src.download_raster --coords "<TopLeftLatitude>_<TopLeftLongitude>_<TopLeftLatitude>_<TopLeftLongitude>_<AGL(feet)>"


## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
