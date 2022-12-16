import sounddevice as sd
import numpy as np
import whisper
import asyncio
import sys
import pyttsx3

# SETTINGS
# the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
MODEL_TYPE="base.en"
# pre-set the language to avoid autodetection
LANGUAGE="English"
# Sample rate
SAMPLE_RATE = 16 * 10**3
# this is the base chunk size the audio is split into in samples. blocksize / sample rate = chunk length in seconds. 
BLOCKSIZE=24678
# should be set to the lowest sample amplitude that the speech in the audio material has
SILENCE_THRESHOLD=400
# number of samples in one buffer that are allowed to be higher than threshold.
SILENCE_RATIO=100

async def inputstream_generator():
	"""Generator that yields blocks of input data as NumPy arrays."""
	q_in = asyncio.Queue() # Infinite length queue
	loop = asyncio.get_event_loop() # Get the running event loop

	def callback(indata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
		"""
		indata: 2D array. But its inner arrays are all single element array.
		"""
		loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

	stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=BLOCKSIZE, callback=callback)
	with stream:
		while True:
			indata, status = await q_in.get()
			yield indata, status

async def process_audio_buffer(model, tts_engine: pyttsx3.Engine):
	global_ndarray = None
	async for indata, status in inputstream_generator():

		indata_flattened = abs(indata.flatten())

		# discard buffers that contain mostly silence
		if(np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size < SILENCE_RATIO):
			# Maybe it's more prompt to run transcribing procedure here
			continue
		if (global_ndarray is not None):
			global_ndarray = np.concatenate((global_ndarray, indata), dtype='int16')
		else:
			global_ndarray = indata
			
		# concatenate buffers if the end of the current buffer is not silent
		if (np.average((indata_flattened[-50:-1])) > SILENCE_THRESHOLD):
			continue
		else:
			local_ndarray = global_ndarray.copy()
			global_ndarray = None
			indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
			result = model.transcribe(indata_transformed, language=LANGUAGE)
			execute_tts(tts_engine, result['text'])
		del local_ndarray
		del indata_flattened

def execute_tts(engine: pyttsx3.Engine, text: str):
	engine.say(text)
	engine.runAndWait()


async def main():
	model = whisper.load_model(MODEL_TYPE)
	tts_engine = pyttsx3.init()
	tts_engine.setProperty('rate', 145)
	print("Loaded whisper model.")
	audio_task = asyncio.create_task(process_audio_buffer(model, tts_engine),name="audio_task")
	print("Created audio_task. Waiting for its completion.")
	await audio_task
	audio_task.cancel()
	print("audio_task completed and canceled.")


if __name__ == "__main__":
	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		sys.exit('\nInterrupted by user')
