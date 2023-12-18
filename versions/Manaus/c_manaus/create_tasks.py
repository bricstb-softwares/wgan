import os

basepath   = os.getcwd()
repo       = os.environ["REPO_BASEPATH"]

# get the current docker image
image_repo = os.environ["SINGULARITY_IMAGES_REPO"]
image_name = os.environ["DOCKER_IMAGE_NAME"]
image_tag  = os.environ["DOCKER_IMAGE_TAG"]
image      = f"{image_repo}/{image_name}_{image_tag}.sif"


# entrypoint after call the singularity image
exec_cmd = f"cd {repo} \n"
exec_cmd+= f"source dev_envs.sh \n"
exec_cmd+= f"cd %JOB_WORKAREA \n"
# command execution
exec_cmd+= f"python {basepath}/run.py -j %IN --username_wandb brics-tb --wandb_taskname %JOB_TASKNAME --is_tb \n"


command = """maestro.py task create \
  -t user.joao.pinto.task.Manaus.c_manaus.wgan_v2_tb.r1 \
  -i {PATH}/jobs \
  --image {IMAGE} \
  --exec "{EXEC}" \
  """



cmd = command.format(PATH=basepath,EXEC=exec_cmd,IMAGE=image)
print(cmd)
#os.system(cmd)


