WORK_DIR=${PWD}
PROJECT=jdt3d
DOCKER_IMAGE=${PROJECT}:eccv_2024
DOCKER_FILE=docker/Dockerfile

DATA_ROOT_LOCAL=./data/nuscenes/
OUTPUT=./job_artifacts

CKPTS_ROOT=${PWD}/ckpts

DOCKER_OPTS = \
	-it \
	--rm \
	-e DISPLAY=${DISPLAY} \
	-v /tmp:/tmp \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /mnt/fsx:/mnt/fsx \
	-v ~/.ssh:/root/.ssh \
	-v ~/.aws:/root/.aws \
	-v ${WORK_DIR}:/workspace/${PROJECT} \
	-v ${CKPTS_ROOT}:/workspace/${PROJECT}/ckpts \
	--shm-size=1G \
	--ipc=host \
	--network=host \
	--pid=host \
	--privileged

DOCKER_BUILD_ARGS = \
	--build-arg AWS_ACCESS_KEY_ID \
	--build-arg AWS_SECRET_ACCESS_KEY \
	--build-arg AWS_DEFAULT_REGION \
	--build-arg WANDB_ENTITY \
	--build-arg WANDB_API_KEY \

docker-build:
	docker image build \
	-f $(DOCKER_FILE) \
	-t $(DOCKER_IMAGE) \
	$(DOCKER_BUILD_ARGS) .


docker-dev:
	docker run \
	--runtime=nvidia \
	--gpus all \
	--name $(PROJECT) \
	-v ${DATA_ROOT_LOCAL}:/workspace/${PROJECT}/data/nuscenes \
	-v ${OUTPUT}:/workspace/${PROJECT}/job_artifacts \
	$(DOCKER_OPTS) \
	$(DOCKER_IMAGE) bash

clean:
	find . -name '"*.pyc' | xargs sudo rm -f && \
	find . -name '__pycache__' | xargs sudo rm -rf