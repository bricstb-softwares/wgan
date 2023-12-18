import os
basepath = os.getcwd()

path = basepath

image = "/home/joao.pinto/public/images/rxwgan_latest.sif"

repo = "/home/joao.pinto/git_repos/brics/rxwgan/"


exec_cmd = f"cd {repo} \n"
exec_cmd+= f"source dev_envs.sh \n"
exec_cmd+= f"cd %JOB_WORKAREA \n"
exec_cmd+= f"python {path}/run.py -j %IN --username_wandb brics-tb --wandb_taskname %JOB_TASKNAME --is_tb \n"


command = """maestro.py task create \
  -t user.joao.pinto.task.Manaus.manaus.wgan_v2_tb.r2 \
  -i {PATH}/jobs \
  --image {IMAGE} \
  --exec "{EXEC}" \
  --skip_test \
  """



cmd = command.format(PATH=path,EXEC=exec_cmd,IMAGE=image)
print(cmd)
os.system(cmd)


