class MicrophoneCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.chunkSize = Math.round(sampleRate * options.processorOptions.chunkSeconds);
    this.limit = Math.round(sampleRate * options.processorOptions.maxDurationSeconds);
    this.buffer = new Float32Array(this.chunkSize);
    this.offset = 0;
    this.total = 0;
    this.active = true;
    this.port.onmessage = (event) => {
      if (event.data.type === "stop") this.stop();
    };
  }

  flush() {
    if (!this.offset) return;
    const chunk = this.buffer.slice(0, this.offset);
    this.port.postMessage({ type: "chunk", buffer: chunk.buffer }, [chunk.buffer]);
    this.offset = 0;
  }

  stop() {
    if (!this.active) return;
    this.active = false;
    this.flush();
    this.port.postMessage({ type: "stopped" });
  }

  process(inputs) {
    if (!this.active) return true;
    const channels = inputs[0];
    if (!channels || !channels.length) return true;
    for (let index = 0; index < channels[0].length; index += 1) {
      let sample = 0;
      for (const channel of channels) sample += channel[index];
      this.buffer[this.offset++] = Math.max(-1, Math.min(1, sample / channels.length));
      this.total += 1;
      if (this.offset === this.chunkSize) this.flush();
      if (this.total >= this.limit) {
        this.stop();
        break;
      }
    }
    return true;
  }
}

registerProcessor("microphone-capture", MicrophoneCapture);
