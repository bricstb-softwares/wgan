



__all__ = [ "Pool", "Slot"]


from pprint import pprint
import argparse
import os, time
import subprocess

from colorama import Back, Fore
import colorama
colorama.init(autoreset=True)



class Slot(object):

  def __init__( self ):
    self.__proc = None
    self.__lock = False

  def lock(self):
    self.__lock=True
  
  def unlock(self):
    self.__lock=False

  def update(self):
    if self.__proc and not self.__proc.poll():
      self.unlock()

  def run(self, command):
    self.__proc = subprocess.Popen(command.split(' '))
    self.lock()

  def isAvailable(self):
    if self.__proc:
      if not self.__proc.poll() is None:
        self.unlock()
    return not self.__lock




class Pool:


  def __init__(self, func, command, maxJobs, files, output ):
    
    self.__files = files
    self.__gen = func
    self.__command = command
    self.__output  = output
    self.__slots = [Slot() for _ in range(maxJobs)]
    self.__outputs = []


  def getAvailable(self):
    for slot in self.__slots:
      if slot.isAvailable():
        return slot
    return None

  
  def busy(self):
    for slot in self.__slots:
      if not slot.isAvailable():
        return True
    return False


  def generate(self):
    f = self.__files.pop()
    idx = len(self.__files)
    output = self.__output + '.' + str(idx)
    self.__outputs.append(output)
    return self.__gen(self.__command, f, output)


  def run( self ):

    total = len(self.__files)

    while len(self.__files) > 0:

      slot = self.getAvailable()
      if slot:
        print(Back.RED + Fore.WHITE + 'running %d/%d'%(len(self.__files), total))
        time.sleep(1)
        command = self.generate()
        slot.run( command )
    
    while self.busy():
      continue


  

