import json, os


# create jobs
def create_jobs(output_path = os.getcwd()+'/jobs'):
  if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
    for test in range(10):
        for sort in range(9):
            d = {'sort':sort,'test':test}
            o = output_path + '/job.test_%d.sort_%d.json'%(test,sort)
            with open(o, 'w') as f:
                json.dump(d, f)


def create_task( task_name, production, info_file, dry_run=False):
  
  local_path  = os.getcwd()
  proj_path   = os.environ["PROJECT_DIR"]
  image_path  = proj_path + '/images/wgan_base.sif'
  config_path = local_path + '/configs/'+ production
  repo_path   = os.environ['REPO_DIR']
  job_path    = os.getcwd()+'/jobs'

  exec_cmd  = f"run_train.py --j %IN -c {config_path}"
  envs      = { 'DATA_DIR':os.environ['DATA_DIR'] }
  binds     = {"/mnt/brics_data":"/mnt/brics_data", "/home":"/home"}
  command = f"""maestro task create \
    -t {task_name} \
    -i {job_path} \
    --exec "{exec_cmd}" \
    --envs "{str(envs)}" \
    --partition gpu \
    --image {image_path} \
    --binds "{str(binds)}" \
    """
  print(command)
  if not dry_run:
    os.system(command)


#
# create production
#


create_jobs()
create_task( 'user.joao.pinto.task.Shenzhen_china.wgan.v2_tb.r1'   , 'china_tb_card.json'    , dry_run=True )
create_task( 'user.joao.pinto.task.Shenzhen_china.wgan.v2_notb.r1' , 'china_notb_card.json'  , dry_run=True )
  

