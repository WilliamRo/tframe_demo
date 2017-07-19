import os


# FLAGs
overwrite = True

epoch = 2
batch_size = 50

shuffle = False
# shuffle = True

# Train
os.system('python ./mnist_deep.py --overwrite {} --epoch {} '
          '--batch_size {} --shuffle {}'.format(
  overwrite, epoch, batch_size, shuffle))