import os
basepath = os.getcwd()

path = basepath

datapath = '/home/jodafons/public/brics_data/Shenzhen/raw/Shenzhen_table_from_raw.csv'

exec_cmd = ". {PATH}/init.sh && "
exec_cmd+= "python {PATH}/run.py -j %IN -i {DATA} -t 0 --username_wandb brics-tb && "
exec_cmd+= ". {PATH}/end.sh"

exec_cmd = exec_cmd.format(PATH=basepath, DATA=datapath)

command = """maestro.py task create \
  -v {PATH} \
  -t user.jodafons.task.Shenzhen.wgan.v2_notb.r1 \
  -i {PATH}/jobs \
  --exec "{EXEC}" \
  """



cmd = command.format(PATH=path,EXEC=exec_cmd)
print(cmd)
os.system(cmd)


