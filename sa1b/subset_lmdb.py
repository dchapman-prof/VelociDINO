import sys
import lmdb
import os


if (len(sys.argv)<5):
    print('usage:')
    print('   python3 subset_lmdb.py  archive.lmdb   in_dir  out_dir   keys.txt')
    sys.exit(1)
archive = sys.argv[1]
in_dir  = sys.argv[2]
out_dir = sys.argv[3]
keys_txt = sys.argv[4]
in_archive  = os.path.join(in_dir, archive)
out_archive = os.path.join(out_dir, archive)
print('archive', archive)
print('in_dir',  in_dir)
print('out_dir', out_dir)
print('keys_txt', keys_txt)
print('in_archive', in_archive)
print('out_archive', out_archive)




print('------')
print(' Read the images')
print('------')
with open(keys_txt, 'r') as f:
        keys_str  = [line.strip() for line in f if line.strip()]
        keys_byte = [key.encode('ascii') for key in keys_str]

print('len(keys_byte)', len(keys_byte))
print('len(keys_str)',  len(keys_str))
n_keys = len(keys_str)

print('------')
print(' Check append order')
print('------')
is_append_order = all(keys_byte[i] <= keys_byte[i+1] for i in range(len(keys_byte)-1))
print('is_append_order: ', is_append_order)

if not is_append_order:
        print('ERROR: not append order, terminating ...')
        sys.exit(1)



print('------')
print(' Open datasets')
print('------')
in_env = lmdb.open(in_archive, readonly=True, lock=False, readahead=False, meminit=False)

map_size = 2 * 1024**4  # 2 Terabytes
out_env = lmdb.open(out_archive, map_size=map_size)

print('------')
print(' For every key')
print('------')

batch_size = 1000

sidx = 0
eidx = 0

while (eidx<n_keys):
	# Start and end indices
	sidx = eidx
	eidx = min(sidx+batch_size, n_keys)
	B = eidx-sidx

	print('----------------')
	print('Read keys:')
	print('      ', keys_str[sidx])
	print('  to  ', keys_str[eidx-1])
	keys = keys_byte[sidx:eidx]
	vals = []
	with in_env.begin(write=False) as in_txn:
		for b in range(B):
			vals.append( in_txn.get(keys[b]) )

	print('Write keys')
	with out_env.begin(write=True) as out_txn:
		for b in range(B):
			out_txn.put(keys[b], vals[b], append=True)



print('Close archives')
in_env.close()
out_env.close()

print('Done!')


