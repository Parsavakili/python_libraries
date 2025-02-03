# -*- coding: utf-8 -*-
import os
import csv
import json
import pickle
import gzip
import tempfile
import shutil
import mmap
import stat
# import fcntl  # Unix-only for file locking
from pathlib import Path

# 1. Opening / create a file
with open('example.txt', 'w') as f:
    f.write("Initial content\n")

# 2. Reading from a file
with open('example.txt', 'r') as f:
    content = f.read()
    print("Original content:", repr(content))

# 3. Writing to a file (overwrites existing content)
with open('example.txt', 'w') as f:
    f.write("New content\nLine 2\nLine 3\n")

# 4. Appending to a file
with open('example.txt', 'a') as f:
    f.write("Appended content\n")

# 5. Closing a file explicitly (not needed with 'with' statement)
f = open('example.txt', 'r')
print("\nFirst line:", f.readline())
f.close()

# 6. File positions
with open('example.txt', 'r+') as f:
    f.seek(10)
    print("\nFrom byte 10:", f.read(5))
    print("Current position:", f.tell())

# 7. Working with binary files
with open('binary.bin', 'wb') as f:
    f.write(b'\x00\x01\x02\x03\x04')
with open('binary.bin', 'rb') as f:
    print("\nBinary content:", f.read())

# 8. File metadata
stats = os.stat('example.txt')
print("\nFile metadata:")
print(f"Size: {stats.st_size} bytes")
print(f"Modified: {stats.st_mtime}")

# 9. Deleting a file
if os.path.exists('binary.bin'):
    os.remove('binary.bin')
    print("\nBinary file deleted")

# 10. Renaming a file
if os.path.exists('example.txt'):
    os.rename('example.txt', 'renamed.txt')

# 11. Checking file existence
print("\nFile exists:", os.path.exists('renamed.txt'))

# 12. Directory operations
os.makedirs('test_dir', exist_ok=True)
print("\nDirectory contents:", os.listdir('.'))
os.rmdir('test_dir')

# 13. Exception handling
try:
    with open('nonexistent.txt', 'r') as f:
        content = f.read()
except FileNotFoundError as e:
    print(f"\nError handled: {e}")

# 14. CSV files
with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Age'])
    writer.writerow(['Alice', 30])
    writer.writerow(['Bob', 25])

with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    print("\nCSV content:")
    for row in reader:
        print(row)

# 15. JSON files
data = {'name': 'Alice', 'skills': ['Python', 'Java']}
with open('data.json', 'w') as f:
    json.dump(data, f)

with open('data.json') as f:
    loaded = json.load(f)
    print("\nJSON content:", loaded)

# 16. Pickle files
data = {'name': 'Alice', 'age': 30}
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)

with open('data.pkl', 'rb') as f:
    loaded = pickle.load(f)
    print("\nPickle content:", loaded)

# 17. File compression
with gzip.open('example.gz', 'wt') as f:
    f.write("Compressed text content")

with gzip.open('example.gz', 'rt') as f:
    print("\nCompressed content:", f.read())

# 18. Temporary files
with tempfile.NamedTemporaryFile(delete=False) as tmp:
    tmp.write(b"Temporary data")
    tmp_name = tmp.name

with open(tmp_name, 'r') as f:
    print("\nTemp file content:", f.read())
os.unlink(tmp_name)

# 19. File iteration
with open('iter.txt', 'w') as f:
    f.write("Line 1\nLine 2\nLine 3")

with open('iter.txt', 'r') as f:
    print("\nFile iteration:")
    for i, line in enumerate(f, 1):
        print(f"Line {i}: {line.strip()}")

# 20. File encoding
with open('utf8.txt', 'w', encoding='utf-8') as f:
    f.write("Café 🍵")

with open('utf8.txt', 'r', encoding='utf-8') as f:
    print("\nEncoded content:", f.read())

# 21. File buffering
with open('buffered.txt', 'w', buffering=1) as f:  # Line buffering
    f.write("Buffered content\n")

# # 22. File locking (Unix systems)
# with open('locked.txt', 'w') as f:
#     fcntl.flock(f, fcntl.LOCK_EX)
#     f.write("Locked content")
#     fcntl.flock(f, fcntl.LOCK_UN)

# 23. Memory-mapped files
# with open('mmap.txt', 'w+b') as f:
#     f.write(b"Memory-mapped content")
#     mmapped = mmap.mmap(f.fileno(), 0)
#     print("\nMemory mapped:", mmapped.read())
#     mmapped.close()

# 24. Pathlib usage
path = Path('pathlib.txt')
path.write_text("Pathlib content")
print("\nPathlib exists:", path.exists())
print("Pathlib content:", path.read_text())

# 25. File permissions
path.chmod(stat.S_IRWXU)
print("\nFile permissions:", oct(path.stat().st_mode)[-3:])

# 26. File ownership (Unix) - requires root privileges
# uid = os.getuid()
# path.chown(uid, -1)

# 27. File truncation
with open('trunc.txt', 'w') as f:
    f.write("This is some long content")
with open('trunc.txt', 'r+') as f:
    f.truncate(10)
    print("\nTruncated content:", f.read())

# 28. File flushing
with open('flush.txt', 'w') as f:
    f.write("Flushed content")
    f.flush()  # Force write to disk

# 29. Low-level file descriptors
fd = os.open('lowlevel.txt', os.O_WRONLY | os.O_CREAT)
os.write(fd, b"Low-level content")
os.close(fd)

# 30. File system operations
shutil.copy('lowlevel.txt', 'copy.txt')
shutil.move('copy.txt', 'moved.txt')
os.makedirs('shutil_dir', exist_ok=True)
shutil.rmtree('shutil_dir')

# Cleanup
for f in ['renamed.txt', 'data.csv', 'data.json', 'data.pkl', 'example.gz',
          'iter.txt', 'utf8.txt', 'buffered.txt', 'locked.txt', 'mmap.txt',
          'pathlib.txt', 'trunc.txt', 'flush.txt', 'lowlevel.txt', 'moved.txt']:
    if os.path.exists(f):
        os.remove(f)