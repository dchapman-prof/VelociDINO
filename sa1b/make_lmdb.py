import sys
import lmdb
import os
from tqdm import tqdm # Optional: pip install tqdm for a progress bar

def create_lmdb_from_folder(folder_path, lmdb_path):
    # 1. Calculate the size
    # For 1.6TB, we set map_size to 2TB to be safe. 
    # This doesn't take up 2TB of disk immediately; it's a 'sparse' limit.
    map_size = 2 * 1024**4  # 2 Terabytes
    
    # 2. Initialize the LMDB environment
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    file_list = os.listdir(folder_path)
    
    # 3. Start the ingestion
    print(f"Starting conversion of {len(file_list)} files...")
    
    with env.begin(write=True) as txn:
        for i, filename in enumerate(tqdm(file_list)):
            file_path = os.path.join(folder_path, filename)
            
            # Only process actual files
            if os.path.isfile(file_path):
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                
                # LMDB keys and values must be bytes
                key = filename.encode('utf-8')
                txn.put(key, file_data)
            
            # 4. Commit every 1,000 files to keep memory stable
            if i % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)

    env.close()
    print("Conversion complete!")



if (len(sys.argv)<3):
    print('usage:')
    print('   python3 make_lmdb.py folder archive.lmdb')
    sys.exit(1)
folder  = sys.argv[1]
archive = sys.argv[2]
print('folder', folder)
print('archive', archive)

# Example usage:
create_lmdb_from_folder(folder, archive)

print('Done!')


