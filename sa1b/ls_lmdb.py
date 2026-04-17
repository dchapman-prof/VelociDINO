import sys
import lmdb

def list_lmdb_contents(lmdb_path, limit=None):
    """
    Opens an LMDB database and prints the keys.
    :param lmdb_path: Path to the LMDB directory or file.
    :param limit: Number of keys to print (useful for 1.6M files!)
    """
    # Open in read-only mode. 
    # 'lock=False' allows you to list while another script is reading.
    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    print(f"--- Contents of {lmdb_path} ---")
    
    count = 0
    with env.begin() as txn:
        cursor = txn.cursor()
        # Iterate through every key-value pair
        for key, value in cursor:
            # Keys are stored as bytes; decode to string for printing
            print(key.decode('utf-8'))
            
            count += 1
            if limit and count >= limit:
                print(f"\n... Stopped at limit of {limit} keys.")
                break
                
    print(f"--- Total keys found: {count} ---")
    env.close()



if (len(sys.argv)<2):
    print('usage:')
    print('   python3 ls_lmdb.py archive.lmdb [limit=None]')
    sys.exit(1)
archive = sys.argv[1]
limit = int(sys.argv[2]) if (len(sys.argv)>2) else None

# Example usage:
list_lmdb_contents(archive, limit=limit)



