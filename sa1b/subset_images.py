import os
import sys
import random

N_subset = 500000

print('------')
print(' Read the images')
print('------')
with open('images.txt', 'r') as f:
	keys_str  = [line.strip() for line in f if line.strip()]
	keys_byte = [key.encode('ascii') for key in keys_str]

print('len(keys_byte)', len(keys_byte))
print('len(keys_str)',  len(keys_str))

print('------')
print(' Check append order')
print('------')
is_append_order = all(keys_byte[i] <= keys_byte[i+1] for i in range(len(keys_byte)-1))
print('is_append_order: ', is_append_order)

if not is_append_order:
	print('ERROR: not append order, terminating ...')
	sys.exit(1)

print('------')
print(' Pick 500000 random indices')
print('------')
indices = list(range(len(keys_byte)))
random.shuffle(indices)
indices = indices[0:N_subset]
indices.sort()

print('indices', indices[0:100], ' and so on . . . ')

print('------')
print(' Write the subset_images')
print('------')

fout = open('images_subset.txt', 'w')
for i in range(N_subset):
	idx = indices[i]
	fout.write(keys_str[idx]+'\n')
fout.close()

print('Done!')


print('Done!')




