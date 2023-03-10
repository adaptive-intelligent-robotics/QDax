SHELL := /bin/bash

# variables
WORK_DIR = $(PWD)
PORT = 8891
DOCKER_RUN_FLAGS = --rm --shm-size=1024m
DOCKER_IMAGE_NAME = instadeep/qdax:$(USER)
TTACH_FOLDERS_FLAGS = -v $(WORK_DIR)/examples:/app/examples -v $(WORK_DIR)/playground:/app/playground

USER_ID = $$(id -u)
GROUP_ID = $$(id -g)

PORT = 8888

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

.PHONY: style
style: clean
	sudo docker run $(DOCKER_RUN_FLAGS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) pre-commit run --all-files

.PHONY: notebook
notebook: build
	echo "Make sure you have properly exposed your VM before, with the gcloud ssh command followed by -- -N -f -L $(PORT):localhost:$(PORT)"
	sudo docker run -p $(PORT):$(PORT) -it $(DOCKER_RUN_FLAGS) \
		$(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(ATTACH_FOLDERS_FLAGS) $(DOCKER_IMAGE_NAME) \
		jupyter notebook --port=$(PORT) --no-browser --ip=0.0.0.0 --allow-root

.PHONY: jupyter_lab
jupyter_lab: build
	echo "Make sure you have properly exposed your VM before, with the gcloud ssh command followed by -- -N -f -L $(PORT):localhost:$(PORT)"
	sudo docker run -p $(PORT):$(PORT) -it $(DOCKER_RUN_FLAGS) \
		$(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(ATTACH_FOLDERS_FLAGS) $(DOCKER_IMAGE_NAME) \
		jupyter lab --port=$(PORT) --no-browser --ip=0.0.0.0 --allow-root
