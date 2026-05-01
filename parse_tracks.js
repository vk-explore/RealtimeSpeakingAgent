const fs = require('fs');
const buffer = fs.readFileSync('avatar.glb');
const magic = buffer.readUInt32LE(0);
if (magic !== 0x46546C67) { console.log('Not a GLB'); process.exit(1); }
const jsonChunkLength = buffer.readUInt32LE(12);
const jsonChunkType = buffer.readUInt32LE(16);
if (jsonChunkType !== 0x4E4F534A) { console.log('No JSON chunk'); process.exit(1); }
const jsonStr = buffer.toString('utf8', 20, 20 + jsonChunkLength);
const gltf = JSON.parse(jsonStr);

let tracks = [];
gltf.animations.forEach(anim => {
    anim.channels.forEach(ch => {
        let nodeName = gltf.nodes[ch.target.node].name;
        tracks.push(`${nodeName}.${ch.target.path}`);
    });
});
console.log(tracks.join('\n'));
