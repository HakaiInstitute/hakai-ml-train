# uav-classif

## Kelp segmentation instructions

### Log in to Hal9000
- Open a terminal and ssh into Hal9000
    - e.g. `ssh username@10.10.1.13`. Enter you password
    - To ssh from Windows, use the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
- To get an account on Hal9000, ask one of the Hakai Tech team, who can create an account for you.
    - Ask for TeamViewer access as well for GUI access to the server. TeamViewer may be required to set a new password and is also useful for checking processing outputs.

### First-time setup
- Install cifs-utils so you can later access NAS files `sudo apt-get install cifs-utils`
- Make mount point for H drive if it doesn’t exist: `sudo mkdir -p /mnt/H`
- Add yourself to the Docker user group: `sudo usermod -aG docker $USER`
    - Log out and log back in for the change to take effect.
    - `docker run hello-world` should print a “Hello World” message to the terminal if everything is working. See https://docs.docker.com/engine/install/linux-postinstall/ if you have any issues.

#### Download the Kelp model weights
- `curl https://hakai-deep-learning-datasets.s3.amazonaws.com/kelp/weights/deeplabv3_kelp_200506.pt > deeplabv3_kelp_200506.pt` will download the weights to your current directory
    - Later, you’ll need to pass the path to the weights to the classification script, so you may want to move them with `mv /from/location /to/location`

#### Clone the GitHub repo
- Install Git if necessary: `sudo apt-get install git`
- Pull the repo: `git clone https://github.com/tayden/uav-classif`

#### Mount the NAS via samba 
- You’ll have to do this after each computer restart. It may be possible for Chris to edit /etc/fstab so this happens automatically for all users.
- Mount the NAS to /mnt/H 
    - e.g. (edit user): `sudo mount -t cifs -o user=firstname.lastname,domain=victoria.hakai.org //10.10.1.50/Geospatial /mnt/H`
    - Follow the password prompts to log in
    - Note: You will probably have to use `sudo` to copy files to/from this directory.

### Classify an image
- Download the image to the local hard drive. It’ll run way faster this way. It's possible to read the file from the NAS, but there will be a network bottleneck.
    - e.g. `cp /mnt/H/path/to/image.* ./local/drive/location/`
- Navigate to the cloned Github repo: e.g. `cd path/to/repo/uav-classif`
- Run the classification script:
    - Using Docker: `bash segment_kelp.sh /path/to/input/file /path/to/desired/output.tif /path/to/weights/deeplabv3-kelp_2000506.pt`
    - Or, using local Python environment:
        - Using the local environment requires installing packages. The easiest way to do this is with conda. In the git repo, run `conda env create`.
        - After installing packages, run `conda activate uav`
        - Now, run `PYTHONPATH=. python segment_kelp.py pred /path/to/input/file /path/to/desired/output.tif /path/to/weights/deeplabv3-kelp_200506.pt`
- Move the output back to the samba server if desired, e.g. `mv /path/to/desired/output.tif /mnt/H/location/of/choice`

### Interpreting model output
- The model outputs a raster image with integer values (0-255). This value should be thresholded to obtain a kelp/not-kelp classification output.
- In most case, setting the threshold such that a pixel being >=128 means kelp should be adequate. 
- If you want to increase kelp recall at the cost of precision, you can lower this threshold (e.g. to 100).

### Tips
#### Killing a job
Press CTRL+C to kill the classification script after it starts if you don’t want to continue.

#### Long running jobs
The easiest way to keep long running jobs active after you log out and start doing other things is to use the screen utility. Before running the segment-kelp.sh script, enter `screen` to start a screen session (press enter at the “welcome” message). Start the command as before, everything will look the same. You can now detach the command so it continues to run in case your ssh session is interrupted. To detach, press (CTL+A+D). The screen is now detached and running in the background. You can resume the session by entering `screen -r`. When the screen session is detached, you can close the terminal or log out of the hal9000 terminal session and not worry about the classification job stopping. Just ssh back into hal9000 and run `screen -r` to check progress, etc.

TLDR;
```
# install screen
sudo apt-get install screen

# start a new session
screen

# execute the script
bash segment_kelp.sh …

# to detach: Press CTRL+A+D
# Script is now running in the background and protected from interuption by
#   logging out, etc.

# to re-attach the session and check progress, etc.
screen -r
```

#### Docker
Docker simply creates a lightweight virtual Linux machine and then executes the script in it. This ensures consistency across environments.
You can inspect the `segment_kelp.sh` file to see exactly what Docker image is being pulled and what volumes are mounted to it.
To access from Hal9000 in the running Docker environment, the data drives must be mapped with the `-v from_location:to_location` flags.
This is all taken care of for you in this script.
