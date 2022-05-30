SHELL := /bin/bash

# variables
WORK_DIR = $(PWD)
DOCKER_RUN_FLAGS = --rm --shm-size=1024m --gpus all
DOCKER_IMAGE_NAME = instadeep/qdax:$(USER)
USER_ID = $$(id -u)
GROUP_ID = $$(id -g)

# Makefile
.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rfv
	find . | grep -E "nul" | xargs rm -rfv

.PHONY: build
build:
	sudo docker build --target dev-image -t $(DOCKER_IMAGE_NAME) -f dev.Dockerfile  . --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID)

.PHONY: dev_container
dev_container: clean build
	sudo docker run -it $(DOCKER_RUN_FLAGS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) /bin/bash

.PHONY: ipynb_container
ipynb_container: clean build
	echo "Make sure you have properly exposed your VM before, with the gcloud ssh command followed by -- -N -f -L 8888:localhost:8888"
	sudo docker run -p 8888:8888 -it $(DOCKER_RUN_FLAGS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root

.PHONY: style
style: clean
	sudo docker run $(DOCKER_RUN_FLAGS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) pre-commit run --all-files
