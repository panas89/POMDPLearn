1. Install Octave 4.2.2 Using Docker

The easiest and cleanest way to use Octave 4.2.2 on a modern system is through a Docker container.
Steps:

    Install Docker (if not already installed):

sudo apt update
sudo apt install docker.io

Run Octave 4.2.2 in Docker:

docker run -it --rm ubuntu:18.04 bash

Inside the container, install Octave 4.2.2:

apt update
apt install octave=4.2.2-1ubuntu1
octave --version