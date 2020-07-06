# uav-classif

## Kelp segmentation instructions

### Log on to Hal9000
- Open a terminal and ssh into Hal9000
    - e.g. `ssh username@10.10.1.13`. Enter your password
    - To ssh from Windows, use the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
- To get an account on Hal9000, ask one of the Hakai Tech team, who can create an account for you.
    - Ask for TeamViewer access as well for GUI access to the server. TeamViewer may be required to set a new password and is also useful for checking processing outputs.

### First-time setup
- Install cifs-utils so you can later access NAS files `sudo apt-get install cifs-utils`
- Make a mount point for H drive if it doesn't exist: `sudo mkdir -p /mnt/H`
- Add yourself to the Docker user group: `sudo usermod -aG docker $USER`
    - Log out and log back in for the change to take effect.
    - `docker run hello-world` should print a “Hello World” message to the terminal if everything is working. See https://docs.docker.com/engine/install/linux-postinstall/ if you have any issues.

#### Download the Kelp model weights
Later, you’ll need to pass the path to the weights to the classification script, so you may want to move them with `mv /from/location /to/location`
##### For Kelp presence segmentation
`curl https://hakai-deep-learning-datasets.s3.amazonaws.com/kelp/weights/deeplabv3_kelp_200704.ckpt > deeplabv3_kelp_200704.ckpt` will download the weights to your current directory
##### For Kelp species segmentation
`curl https://hakai-deep-learning-datasets.s3.amazonaws.com/kelp_species/weights/deeplabv3_kelp_species_200611.ckpt > deeplabv3_kelp_species_200611.ckpt` will download the weights to your current directory

#### Clone the GitHub repo
- Install Git if necessary: `sudo apt-get install git`
- Clone the repo: `git clone https://github.com/tayden/uav-classif`

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
- Update the GitHub repo if you haven't lately. This ensures you have the latest version of the code and these instructions.
    - Run `git pull`
- Run the classification script:
    - Using Docker: 
        - For species: `bash segment_kelp_species.sh /path/to/input/file /path/to/desired/output.tif /path/to/weights/deeplabv3_kelp_species_200611.ckpt`
        - For presence/absence: `bash segment_kelp.sh /path/to/input/file /path/to/desired/output.tif /path/to/weights/deeplabv3_kelp_200704.ckpt`
    - Or, using a local Python environment:
        - Using the local environment requires installing packages. The easiest way to do this is with conda. In the git repo, run `conda env create`.
        - After installing packages, run `conda activate uav`
        - Now, run `PYTHONPATH=. python segment_kelp_species.py pred /path/to/input/file /path/to/desired/output.tif /path/to/weights/deeplabv3_kelp_species_200611.ckpt`
- Move the output back to the samba server if desired, e.g. `mv /path/to/desired/output.tif /mnt/H/location/of/choice`

### Interpreting the model output
- The model outputs a raster image with integer values (0-255). Each value is a class label.
    - For presence/absence, 1 is kelp, 0 is not kelp
    - For species, 0 is not kelp, 1 is macro, 2 is nereo.

### Tips
#### Killing a job
Press CTRL+C to kill the classification script after it starts if you don’t want to continue.

#### Long-running jobs
The easiest way to keep long-running jobs active after you log out and start doing other things is to use the screen utility. Before running the segment-kelp.sh script, enter `screen` to start a screen session (press enter at the “welcome” message). Start the command as before, everything will look the same. You can now detach the command so it continues to run in case your ssh session is interrupted. To detach, press (CTL+A+D). The screen is now detached and running in the background. You can resume the session by entering `screen -r`. When the screen session is detached, you can close the terminal or log out of the hal9000 terminal session and not worry about the classification job stopping. Just ssh back into hal9000 and run `screen -r` to check progress, etc.

TLDR;
```
# install screen
sudo apt-get install screen

# start a new session
screen

# execute the script
bash segment_kelp_species.sh …

# to detach: Press CTRL+A+D
# Script is now running in the background and protected from interruption by
#   logging out, etc.

# to re-attach the session and check progress, etc.
screen -r
```

#### Queuing multiple jobs
If you have a lot of files to process, it is much more convenient to be able to dynamically add them to a queue such that the processing happens
sequentially and you don't need to supervise each segmentation run so you know when you can start the next job. The downloaded GitHub repo contains
a toolset for doing just this in the "job_queue" directory.

##### Processing jobs in the queue
There is a script that is essentially a loop that checks the queue for new jobs and processes them in the order they were submitted. This script is called `job_queue/run.py`.
To start the queue, run `python run.py`. You may want to run this in a screen session so it doesn't get killed when you log out. See the "Long-running jobs" section of this Readme 
for details on the screen util. 

##### Adding, removing, and manipulation of the job queue
Also in the `job_queue` directory, there is another script called "job.py" useful for manipulating the job queue.
Documentation for all of the functions available can be printed to the terminal using `python job.py --help`.

The most important command for you is likely to be `python job.py add /path/to/in_image.tif /path/to/where/you/want/the/output.tif /path/to/the/weight/file`

There are other commands to restart jobs, remove jobs, list jobs and the status, and mark jobs as complete. Use `python job.py --help` for all the details of each.

#### Docker
Docker simply creates a lightweight virtual Linux machine and then executes the script in it. This ensures consistency across environments.
You can inspect the `segment_kelp_species.sh` file to see exactly what Docker image is being pulled and what volumes are mounted to it.
To access from Hal9000 in the running Docker environment, the data drives must be mapped with the `-v from_location:to_location` flags.
This is all taken care of for you in this script.
