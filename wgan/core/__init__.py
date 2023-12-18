__all__ = ['declare_property', 
           'allow_tf_growth',
           'MSG_INFO',
           'MSG_WARNING',
           'MSG_ERROR',
           'MSG_FATAL']

from colorama import init, Back, Fore

init(autoreset=True)



#
# declare property
#
def declare_property( cls, kw, name, value , private=False):
  atribute = ('__' + name ) if private else name
  if name in kw.keys():
    setattr(cls,atribute, kw[name])
  else:
    setattr(cls,atribute, value)


def allow_tf_growth():
  import os
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'


def MSG_INFO(message):
  print(Fore.GREEN + message)

def MSG_WARNING(message):
  print(Fore.YELLOW + message)

def MSG_ERROR(message):
  print(Fore.RED + message)

def MSG_FATAL(message):
  print(Back.RED + Fore.WHITE + message)



from . import stats
__all__.extend( stats.__all__ )
from .stats import *

from . import metrics
__all__.extend( metrics.__all__ )
from .metrics import *

from . import stratified_kfold
__all__.extend( stratified_kfold.__all__ )
from .stratified_kfold import *

from . import MultiProcessing
__all__.extend( MultiProcessing.__all__ )
from .MultiProcessing import *