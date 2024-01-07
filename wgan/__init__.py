__all__ = ["DATA_DIR", "declare_property", "allow_tf_growth"]

import os

DATA_DIR = os.environ["DATA_DIR"]


def declare_property( cls, kw, name, value , private=False):
  atribute = ('__' + name ) if private else name
  if name in kw.keys():
    setattr(cls,atribute, kw[name])
  else:
    setattr(cls,atribute, value)


def allow_tf_growth():
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'


from . import core
__all__.extend( core.__all__ )
from .core import *

from . import wgangp
__all__.extend( wgangp.__all__ )
from .wgangp import *

from . import models
__all__.extend( models.__all__ )
from .models import *

