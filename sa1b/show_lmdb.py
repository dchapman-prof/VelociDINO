import sys
import lmdb
import io
from PIL import Image

def display_image_from_lmdb(lmdb_path, image_key):
	# 1. Open the environment in read-only mode
	env = lmdb.open(lmdb_path, readonly=True, lock=False)

	with env.begin() as txn:
		# 2. Fetch the raw binary data using the key
		# Remember: Keys must be bytes!
		raw_data = txn.get(image_key.encode('utf-8'))	

		print('type(raw_data)', type(raw_data))

		if raw_data is None:
			print(f"Error: Key '{image_key}' not found.")
			return

		# 3. Convert the raw bytes into a file-like object in memory
		image_stream = io.BytesIO(raw_data)

		# 4. Open the image and display it using your OS's default viewer
		img = Image.open(image_stream)
		print(f"Displaying {image_key} ({img.size[0]}x{img.size[1]})")
		img.show()

	env.close()

if (len(sys.argv)<2):
	print('usage:')
	print('   python3 show_lmdb.py my_img.jpg')
	sys.exit(1)
imgname = sys.argv[1]
print('open images.lmdb ', imgname)

# Example Usage:
display_image_from_lmdb('images.lmdb', imgname)

print('Done!')
