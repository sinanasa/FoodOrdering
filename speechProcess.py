import queue
import re
import sys
import time
from GmailSender import GmailSender
import threading

# from google.cloud import speech
import pyaudio

import json
from google.cloud import texttospeech
from playsound import playsound
from RestaurantOrder import RestaurantOrder
from llama_index.core.llms import ChatMessage, MessageRole

# Audio recording parameters
STREAMING_LIMIT = 240000  # 4 minutes
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"


def get_current_time() -> int:
    """Return Current Time in MS.

    Returns:
        int: Current Time in MS.
    """

    return int(round(time.time() * 1000))


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(
        self: object,
        rate: int,
        chunk_size: int,
    ) -> None:
        """Creates a resumable microphone stream.

        Args:
        self: The class instance.
        rate: The audio file's sampling rate.
        chunk_size: The audio file's chunk size.

        returns: None
        """
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

    def __enter__(self: object) -> object:
        """Opens the stream.

        Args:
        self: The class instance.

        returns: None
        """
        self.closed = False
        return self

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> object:
        """Closes the stream and releases resources.

        Args:
        self: The class instance.
        type: The exception type.
        value: The exception value.
        traceback: The exception traceback.

        returns: None
        """
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
        self: object,
        in_data: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Continuously collect data from the audio stream, into the buffer.

        Args:
        self: The class instance.
        in_data: The audio data as a bytes object.
        args: Additional arguments.
        kwargs: Additional arguments.

        returns: None
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self: object) -> object:
        """Stream Audio from microphone to API and to local buffer

        Args:
            self: The class instance.

        returns:
            The data from the audio stream.
        """
        while not self.closed:
            data = []

            if self.new_stream and self.last_audio_input:
                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:
                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round(
                        (self.final_request_end_time - self.bridging_offset)
                        / chunk_time
                    )

                    self.bridging_offset = round(
                        (len(self.last_audio_input) - chunks_from_ms) * chunk_time
                    )

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b"".join(data)


def listen_print_loop(responses: object, stream: object, chatService) -> None:
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.

    Arg:
        responses: The responses returned from the API.
        stream: The audio stream to be processed.
    """

    # Instantiates a tts client
    # client = texttospeech.TextToSpeechClient()

    # init gmailSender
    gmailSender = GmailSender('algotrader506@gmail.com', 'sevm kgqb wbpo pcrr')

    for response in responses:
        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        result_seconds = 0
        result_micros = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.microseconds:
            result_micros = result.result_end_time.microseconds

        stream.result_end_time = int((result_seconds * 1000) + (result_micros / 1000))

        corrected_time = (
            stream.result_end_time
            - stream.bridging_offset
            + (STREAMING_LIMIT * stream.restart_counter)
        )
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.

        if result.is_final:
            sys.stdout.write(GREEN)
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\n")



            # print("Call chat with text {}".format(transcript))
            # print(chatService.chatAway(transcript))
#############
            try:
                chatResponse = chatService.chatAway(transcript)
                response_str = str(chatResponse).replace('\n','')

                #check if the reponse is proper json
                # if yes process
                # else reply to customer

                json_string = chatService.extract_json(response_str)

                if json_string:
                    # Parse the JSON message
                    try:
                        order_data = json.loads(json_string)
                        print("Extracted JSON data:")
                        print(json.dumps(order_data, indent=2))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                else:
                    # print("No JSON data found in the text.")
                    raise Exception

                # order_data = json.loads(response_str)

                # One option was to maintain a separate order list by processing each response as json message
                # if 'general' in jsonResponse['request_type'].replace('_', ' '):
                #     print(f"Chatbot: {jsonResponse['response']}")
                # elif 'add' in jsonResponse['request_type'].replace('_', ' '):
                #     order.add_item(jsonResponse['menu_item_ordered'], jsonResponse['quantity'], jsonResponse['size'])
                #     print(f"Chatbot: Added {jsonResponse['menu_item_ordered']} to the order")
                # elif 'completed' in jsonResponse['request_type'].replace('_', ' '):
                #     self.setPromptOrderTaking3(self.query_engine, order.order_summary())
                # if len(order) == 0:
                #     order.append(jsonResponse)

                # Better option is to receive a json when customer order finalized....
                # Initialize RestaurantOrder object here and send to POS/email. etc.

                # Extract and process the menu items ordered
                items_ordered = order_data["menu_items_ordered"]
                total_price = 0

                print("menu_items_ordered")
                for item in items_ordered:
                    item_name = item["item"]
                    item_size = item["size"]
                    item_quantity = item["quantity"]
                    item_price = float(item["price"].replace('$', ''))  # Convert price to float
                    total_price += item_price*item_quantity
                    print(f"{item_quantity} {item_name} ({item_size}) - ${item_price:.2f}")

                # Extract total price from the JSON and compare with calculated total price
                json_total_price = float(order_data["total_price"].replace('$', ''))

                print(f"\nCalculated Total Price: ${total_price:.2f}")
                print(f"Total Price from JSON: ${json_total_price:.2f}")

                # Check if the calculated total matches the one from the JSON
                if total_price == json_total_price:
                    print("The total price matches!")
                else:
                    print("Warning: The total price does not match!")

                # Print delivery information
                if order_data["pickup_or_delivery"] != "pickup":
                    print(f"Delivery to: {order_data['address']}")
                else:
                    print("Pickup order")

                # Send an email with order content (use the same json to create order in pos)
                gmailSender.send_email("sinan.asa@me.com", "Customer Order 0001", str(json_string))

            except:
                print(f"Chatbot: {chatResponse}")
################################
                # # Set the text input to be synthesized
                # synthesis_input = texttospeech.SynthesisInput(text=response_str)
                #
                # # Build the voice request, select the language code ("en-US") and the ssml
                # # voice gender ("neutral")
                # voice = texttospeech.VoiceSelectionParams(
                #     language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                # )
                #
                # # Select the type of audio file you want returned
                # audio_config = texttospeech.AudioConfig(
                #     audio_encoding=texttospeech.AudioEncoding.MP3
                # )
                #
                # # Perform the text-to-speech request on the text input with the selected
                # # voice parameters and audio file type
                # response = client.synthesize_speech(
                #     input=synthesis_input, voice=voice, audio_config=audio_config
                # )
                #
                #
                # # # The response's audio_content is binary.
                # with open("output.mp3", "wb") as out:
                #     # Write the response to the output file.
                #     out.write(response.audio_content)
                #     print('Audio content written to file "output.mp3"')
                #
                # playsound("output.mp3")
                #
                # # start_time = time.time()
                # # for resp in responses:
                # #     # pass  # This will iterate to the end of the iterator
                # #     if resp.results:
                # #         print(resp.results[0].alternatives[0].transcript)
                # #     # Calculate elapsed time
                # #     end_time = time.time()
                # #     elapsed_time = end_time - start_time
                # #     print(f"Time taken: {elapsed_time:.6f} seconds")
                # #     if (elapsed_time>3):
                # #         break
                # # print('Mic stream has been reseted... No this did not work... not resetting the mic')
                #
########################### End
######################################

            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                sys.stdout.write(YELLOW)
                sys.stdout.write("Exiting...\n")
                stream.closed = True
                break
        else:
            sys.stdout.write(RED)
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\r")

            stream.last_transcript_was_final = False
