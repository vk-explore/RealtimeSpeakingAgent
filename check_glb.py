import struct, json

with open('avatar.glb', 'rb') as f:
    magic = f.read(4)
    print(f"Magic: {magic}")
    version = struct.unpack('<I', f.read(4))[0]
    length = struct.unpack('<I', f.read(4))[0]
    print(f"Version: {version}, Total length: {length}")
    chunk_len = struct.unpack('<I', f.read(4))[0]
    chunk_type = f.read(4)
    print(f"JSON chunk: {chunk_len} bytes, type: {chunk_type}")
    json_data = json.loads(f.read(chunk_len))

# Dump all mesh info
print(f"\nMeshes: {len(json_data.get('meshes', []))}")
for i, mesh in enumerate(json_data.get('meshes', [])):
    name = mesh.get('name', '?')
    prims = mesh.get('primitives', [])
    extras = mesh.get('extras', {})
    weights = mesh.get('weights', [])
    print(f"\n  Mesh {i}: '{name}', {len(prims)} primitives, weights={len(weights)}")
    if extras:
        print(f"    extras keys: {list(extras.keys())}")
    for j, prim in enumerate(prims):
        targets = prim.get('targets', [])
        prim_extras = prim.get('extras', {})
        print(f"    Prim {j}: targets={len(targets)}, keys={list(prim.keys())}")
        if prim_extras:
            print(f"      prim extras keys: {list(prim_extras.keys())}")

# Check nodes for morph target weights
print(f"\nNodes: {len(json_data.get('nodes', []))}")
for i, node in enumerate(json_data.get('nodes', [])):
    if 'weights' in node or 'extras' in node:
        name = node.get('name', '?')
        weights = node.get('weights', [])
        extras = node.get('extras', {})
        if weights or extras:
            print(f"  Node {i}: '{name}', weights={len(weights)}")
            if extras:
                print(f"    extras keys: {list(extras.keys())}")
                for k, v in extras.items():
                    if isinstance(v, list) and len(v) > 0:
                        print(f"    {k}: [{v[0]}, ...] ({len(v)} items)")

# Search entire JSON for targetNames or morphTargets
json_str = json.dumps(json_data)
for keyword in ['targetNames', 'morphTarget', 'blendShape', 'POSITION_0', 'targets']:
    if keyword.lower() in json_str.lower():
        print(f"\nFound '{keyword}' in JSON")
