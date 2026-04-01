"""
Voice AEC module — macOS Voice Processing I/O AudioUnit.

Uses Apple's kAudioUnitSubType_VoiceProcessingIO which provides
hardware-accelerated echo cancellation, noise suppression, and AGC.
This is the same AEC that FaceTime and Siri use.

The AudioUnit handles both mic input and speaker output in one unit,
so the AEC has direct access to the speaker reference signal — no
manual delay estimation needed.

macOS only.
"""

import ctypes
import ctypes.util
import struct
import asyncio
import threading

# ── Load native frameworks ─────────────────────────────────────────
_au_lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("AudioUnit"))
_at_lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("AudioToolbox"))

# ── Core Audio types ───────────────────────────────────────────────
AudioUnit = ctypes.c_void_p
AudioComponentInstance = ctypes.c_void_p
OSStatus = ctypes.c_int32


class AudioComponentDescription(ctypes.Structure):
    _fields_ = [
        ("componentType", ctypes.c_uint32),
        ("componentSubType", ctypes.c_uint32),
        ("componentManufacturer", ctypes.c_uint32),
        ("componentFlags", ctypes.c_uint32),
        ("componentFlagsMask", ctypes.c_uint32),
    ]


class AudioStreamBasicDescription(ctypes.Structure):
    _fields_ = [
        ("mSampleRate", ctypes.c_double),
        ("mFormatID", ctypes.c_uint32),
        ("mFormatFlags", ctypes.c_uint32),
        ("mBytesPerPacket", ctypes.c_uint32),
        ("mFramesPerPacket", ctypes.c_uint32),
        ("mBytesPerFrame", ctypes.c_uint32),
        ("mChannelsPerFrame", ctypes.c_uint32),
        ("mBitsPerChannel", ctypes.c_uint32),
        ("mReserved", ctypes.c_uint32),
    ]


class AudioBuffer(ctypes.Structure):
    _fields_ = [
        ("mNumberChannels", ctypes.c_uint32),
        ("mDataByteSize", ctypes.c_uint32),
        ("mData", ctypes.c_void_p),
    ]


class AudioBufferList(ctypes.Structure):
    _fields_ = [
        ("mNumberBuffers", ctypes.c_uint32),
        ("mBuffers", AudioBuffer * 1),
    ]


# Render callback signature
AURenderCallback = ctypes.CFUNCTYPE(
    OSStatus,
    ctypes.c_void_p,  # inRefCon
    ctypes.POINTER(ctypes.c_uint32),  # ioActionFlags
    ctypes.c_void_p,  # inTimeStamp
    ctypes.c_uint32,  # inBusNumber
    ctypes.c_uint32,  # inNumberFrames
    ctypes.POINTER(AudioBufferList),  # ioData
)


class AURenderCallbackStruct(ctypes.Structure):
    _fields_ = [
        ("inputProc", AURenderCallback),
        ("inputProcRefCon", ctypes.c_void_p),
    ]


# ── Constants ──────────────────────────────────────────────────────
def _fourcc(s):
    return struct.unpack(">I", s.encode("ascii"))[0]

kAudioUnitType_Output = _fourcc("auou")
kAudioUnitSubType_VoiceProcessingIO = _fourcc("vpio")
kAudioUnitManufacturer_Apple = _fourcc("appl")

kAudioUnitScope_Global = 0
kAudioUnitScope_Input = 1
kAudioUnitScope_Output = 2

kAudioUnitProperty_StreamFormat = 8
kAudioUnitProperty_SetRenderCallback = 23
kAudioOutputUnitProperty_EnableIO = 2003
kAudioOutputUnitProperty_SetInputCallback = 2005

kAudioFormatLinearPCM = _fourcc("lpcm")
kLinearPCMFormatFlagIsSignedInteger = 1 << 2
kLinearPCMFormatFlagIsPacked = 1 << 3

# ── C function signatures ─────────────────────────────────────────
AudioComponentFindNext = _at_lib.AudioComponentFindNext
AudioComponentFindNext.restype = ctypes.c_void_p
AudioComponentFindNext.argtypes = [ctypes.c_void_p, ctypes.POINTER(AudioComponentDescription)]

AudioComponentInstanceNew = _at_lib.AudioComponentInstanceNew
AudioComponentInstanceNew.restype = OSStatus
AudioComponentInstanceNew.argtypes = [ctypes.c_void_p, ctypes.POINTER(AudioComponentInstance)]

AudioUnitInitialize = _au_lib.AudioUnitInitialize
AudioUnitInitialize.restype = OSStatus
AudioUnitInitialize.argtypes = [AudioUnit]

AudioOutputUnitStart = _au_lib.AudioOutputUnitStart
AudioOutputUnitStart.restype = OSStatus
AudioOutputUnitStart.argtypes = [AudioUnit]

AudioOutputUnitStop = _au_lib.AudioOutputUnitStop
AudioOutputUnitStop.restype = OSStatus
AudioOutputUnitStop.argtypes = [AudioUnit]

AudioUnitUninitialize = _au_lib.AudioUnitUninitialize
AudioUnitUninitialize.restype = OSStatus
AudioUnitUninitialize.argtypes = [AudioUnit]

AudioComponentInstanceDispose = _at_lib.AudioComponentInstanceDispose
AudioComponentInstanceDispose.restype = OSStatus
AudioComponentInstanceDispose.argtypes = [AudioUnit]

AudioUnitSetProperty = _au_lib.AudioUnitSetProperty
AudioUnitSetProperty.restype = OSStatus
AudioUnitSetProperty.argtypes = [
    AudioUnit, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_void_p, ctypes.c_uint32,
]

AudioUnitRender = _au_lib.AudioUnitRender
AudioUnitRender.restype = OSStatus
AudioUnitRender.argtypes = [
    AudioUnit, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(AudioBufferList),
]


def _check(status, msg=""):
    if status != 0:
        raise RuntimeError(f"CoreAudio error {status}: {msg}")


class VoiceProcessor:
    """
    macOS Voice Processing I/O AudioUnit.

    Provides mic input with hardware AEC and speaker output in one unit.
    Mic audio (echo-cancelled) is delivered via an async queue.
    Speaker audio is fed via feed_speaker_audio().
    """

    def __init__(self, speaker_sample_rate: int = 24000, mic_output_rate: int = 16000, channels: int = 1):
        self.speaker_sample_rate = speaker_sample_rate
        self.mic_output_rate = mic_output_rate
        self.channels = channels
        self._unit = AudioComponentInstance()
        self._started = False

        # Queues
        self.mic_queue: asyncio.Queue[bytes] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Speaker buffer (ring buffer for render callback)
        self._speaker_lock = threading.Lock()
        self._speaker_buf = bytearray()

        self._setup()

    def _setup(self):
        # Find VPIO component
        desc = AudioComponentDescription(
            componentType=kAudioUnitType_Output,
            componentSubType=kAudioUnitSubType_VoiceProcessingIO,
            componentManufacturer=kAudioUnitManufacturer_Apple,
            componentFlags=0,
            componentFlagsMask=0,
        )
        comp = AudioComponentFindNext(None, ctypes.byref(desc))
        if not comp:
            raise RuntimeError("VoiceProcessingIO AudioUnit not found")

        _check(AudioComponentInstanceNew(comp, ctypes.byref(self._unit)),
               "AudioComponentInstanceNew")

        unit = self._unit

        # Enable input (mic) on bus 1
        enable = ctypes.c_uint32(1)
        _check(AudioUnitSetProperty(
            unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1,
            ctypes.byref(enable), ctypes.sizeof(enable),
        ), "Enable mic input")

        # Enable output (speaker) on bus 0
        _check(AudioUnitSetProperty(
            unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0,
            ctypes.byref(enable), ctypes.sizeof(enable),
        ), "Enable speaker output")

        def _make_fmt(rate):
            return AudioStreamBasicDescription(
                mSampleRate=float(rate),
                mFormatID=kAudioFormatLinearPCM,
                mFormatFlags=kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked,
                mBytesPerPacket=2 * self.channels,
                mFramesPerPacket=1,
                mBytesPerFrame=2 * self.channels,
                mChannelsPerFrame=self.channels,
                mBitsPerChannel=16,
                mReserved=0,
            )

        # Speaker bus (0) at speaker_sample_rate (24kHz)
        spk_fmt = _make_fmt(self.speaker_sample_rate)
        _check(AudioUnitSetProperty(
            unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0,
            ctypes.byref(spk_fmt), ctypes.sizeof(spk_fmt),
        ), "Set speaker format")

        # Mic bus (1) also at speaker_sample_rate so VPIO AEC works
        # (both buses must match for echo cancellation)
        mic_fmt = _make_fmt(self.speaker_sample_rate)
        _check(AudioUnitSetProperty(
            unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1,
            ctypes.byref(mic_fmt), ctypes.sizeof(mic_fmt),
        ), "Set mic format")

        # Set render callback for speaker output (bus 0)
        # Must keep reference alive
        self._render_cb = AURenderCallback(self._speaker_render_callback)
        cb_struct = AURenderCallbackStruct(
            inputProc=self._render_cb,
            inputProcRefCon=None,
        )
        _check(AudioUnitSetProperty(
            unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0,
            ctypes.byref(cb_struct), ctypes.sizeof(cb_struct),
        ), "Set render callback")

        # Set input callback (notification that mic data is available)
        self._input_cb = AURenderCallback(self._mic_input_callback)
        input_cb_struct = AURenderCallbackStruct(
            inputProc=self._input_cb,
            inputProcRefCon=None,
        )
        _check(AudioUnitSetProperty(
            unit, kAudioOutputUnitProperty_SetInputCallback,
            kAudioUnitScope_Global, 0,
            ctypes.byref(input_cb_struct), ctypes.sizeof(input_cb_struct),
        ), "Set input callback")

        _check(AudioUnitInitialize(unit), "AudioUnitInitialize")

    @staticmethod
    def _resample_linear(pcm_bytes, from_rate, to_rate):
        """Simple linear interpolation resampler for 16-bit mono PCM."""
        if from_rate == to_rate:
            return pcm_bytes
        n_in = len(pcm_bytes) // 2
        samples_in = struct.unpack(f"<{n_in}h", pcm_bytes)
        ratio = from_rate / to_rate
        n_out = int(n_in / ratio)
        samples_out = []
        for i in range(n_out):
            pos = i * ratio
            idx = int(pos)
            frac = pos - idx
            if idx + 1 < n_in:
                val = samples_in[idx] * (1 - frac) + samples_in[idx + 1] * frac
            else:
                val = samples_in[min(idx, n_in - 1)]
            samples_out.append(max(-32768, min(32767, int(val))))
        return struct.pack(f"<{n_out}h", *samples_out)

    def _speaker_render_callback(self, ref_con, flags, timestamp, bus, n_frames, buf_list):
        """Called by CoreAudio when it needs speaker samples."""
        needed = n_frames * 2 * self.channels
        with self._speaker_lock:
            if len(self._speaker_buf) >= needed:
                data = bytes(self._speaker_buf[:needed])
                del self._speaker_buf[:needed]
            else:
                # Pad with silence
                available = bytes(self._speaker_buf)
                del self._speaker_buf[:]
                data = available + b"\x00" * (needed - len(available))

        buf = buf_list.contents.mBuffers[0]
        ctypes.memmove(buf.mData, data, needed)
        buf.mDataByteSize = needed
        return 0

    def _mic_input_callback(self, ref_con, flags, timestamp, bus, n_frames, buf_list_ignored):
        """Called when mic data is available. We render (pull) it from bus 1."""
        byte_size = n_frames * 2 * self.channels
        buf = AudioBuffer(
            mNumberChannels=self.channels,
            mDataByteSize=byte_size,
            mData=0,
        )
        data_buf = (ctypes.c_char * byte_size)()
        buf.mData = ctypes.cast(data_buf, ctypes.c_void_p)

        abl = AudioBufferList(mNumberBuffers=1)
        abl.mBuffers[0] = buf

        action_flags = ctypes.c_uint32(0)
        status = AudioUnitRender(
            self._unit, ctypes.byref(action_flags), timestamp, 1, n_frames,
            ctypes.byref(abl),
        )
        if status == 0:
            pcm = bytes(data_buf)
            # Resample from VPIO rate (24kHz) down to Gemini input rate (16kHz)
            if self.speaker_sample_rate != self.mic_output_rate:
                pcm = self._resample_linear(pcm, self.speaker_sample_rate, self.mic_output_rate)
            if self.mic_queue is not None and self._loop is not None:
                self._loop.call_soon_threadsafe(self.mic_queue.put_nowait, pcm)

        return 0

    def start(self, loop: asyncio.AbstractEventLoop, mic_queue: asyncio.Queue):
        self._loop = loop
        self.mic_queue = mic_queue
        _check(AudioOutputUnitStart(self._unit), "AudioOutputUnitStart")
        self._started = True
        print("[VoiceProcessor] Started macOS Voice Processing I/O (hardware AEC)")

    def feed_speaker_audio(self, pcm_bytes: bytes):
        """Feed agent audio for playback through the VPIO unit."""
        with self._speaker_lock:
            self._speaker_buf.extend(pcm_bytes)

    def flush_speaker(self):
        """Clear buffered speaker audio (e.g. on interruption)."""
        with self._speaker_lock:
            self._speaker_buf.clear()

    def stop(self):
        if self._started:
            AudioOutputUnitStop(self._unit)
            self._started = False
        AudioUnitUninitialize(self._unit)
        AudioComponentInstanceDispose(self._unit)
