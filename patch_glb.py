import struct, json, shutil

def read_glb(path):
    with open(path, 'rb') as f:
        header = f.read(12)
        magic, version, total_len = struct.unpack('<4sII', header)
        json_chunk_len = struct.unpack('<I', f.read(4))[0]
        json_chunk_type = f.read(4)
        json_bytes = f.read(json_chunk_len)
        bin_chunk_len = struct.unpack('<I', f.read(4))[0]
        bin_chunk_type = f.read(4)
        bin_bytes = f.read(bin_chunk_len)
        return json.loads(json_bytes), bin_bytes, json_chunk_type, bin_chunk_type

def write_glb(path, json_data, bin_bytes, json_chunk_type, bin_chunk_type):
    json_bytes = json.dumps(json_data, separators=(',', ':')).encode('utf-8')
    # Pad JSON to 4-byte alignment
    while len(json_bytes) % 4 != 0:
        json_bytes += b' '
    # Pad BIN to 4-byte alignment
    bin_pad = b''
    while (len(bin_bytes) + len(bin_pad)) % 4 != 0:
        bin_pad += b'\x00'
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes) + len(bin_pad)
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, total))
        f.write(struct.pack('<I', len(json_bytes)))
        f.write(json_chunk_type)
        f.write(json_bytes)
        f.write(struct.pack('<I', len(bin_bytes) + len(bin_pad)))
        f.write(bin_chunk_type)
        f.write(bin_bytes)
        f.write(bin_pad)

# Read both
old_json, _, _, _ = read_glb('avatar_old.glb')
new_json, new_bin, jct, bct = read_glb('avatar.glb')

# Build map: target_count -> targetNames from old
old_names = {}
for mesh in old_json['meshes']:
    extras = mesh.get('extras', {})
    names = extras.get('targetNames', [])
    if names:
        old_names[len(names)] = names

print('Old name sets by count:', {k: v[:3] for k, v in old_names.items()})

# Apply to new meshes
patched = 0
for mesh in new_json['meshes']:
    prims = mesh.get('primitives', [])
    n_targets = len(prims[0].get('targets', [])) if prims else 0
    if n_targets > 0 and n_targets in old_names:
        if 'extras' not in mesh:
            mesh['extras'] = {}
        mesh['extras']['targetNames'] = old_names[n_targets]
        print(f'Patched mesh "{mesh.get("name","?")}" with {n_targets} names')
        patched += 1

print(f'Patched {patched} meshes total')

# Backup and write
shutil.copy('avatar.glb', 'avatar_backup.glb')
write_glb('avatar.glb', new_json, new_bin, jct, bct)
print('Done! avatar.glb patched, backup saved as avatar_backup.glb')
