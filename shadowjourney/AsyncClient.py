import aiohttp
import asyncio
import json
from .exceptions import *
from .responses.chat import ChatCompletion, Choice, ChatCompletionMessage, CompletionUsage, CompletionTokensDetails, PromptTokensDetails
from .responses import chat_stream
from .responses.images import *
from .responses.transcription import *
from .responses.embedding import *
from urllib.parse import urlparse

class AsyncAPI:
    def __init__(self, api_key, base_url="https://shadowjourney.xyz/v1"):
        self.api_key = api_key
        self.url = base_url if self.validate_url(base_url) else "https://shadowjourney.xyz/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }   
        self.chat = self.Chat(self)
        self.images = self.Images(self)

    def validate_url(self, url):
        parsed_url = urlparse(url)
        return all([parsed_url.scheme, parsed_url.netloc])

    def handle_http_error(self, status_code, response_text):
        """
        Handles HTTP errors and raises custom exceptions based on status codes.
        """
        if status_code == 400:
            raise InvalidRequestError(f"Bad Request: {response_text}")
        elif status_code == 401:
            raise AuthenticationError(f"Authentication Failed: {response_text}")
        elif status_code == 403:
            raise AccessDeniedError(f"Access Denied: {response_text}")
        elif status_code == 404:
            raise ResourceNotFoundError(f"Resource Not Found: {response_text}")
        elif status_code == 429:
            raise RateLimitError(f"Rate limit reached!: {response_text}")
        elif status_code == 500:
            raise ServerError(f"Server Error: {response_text}")
        else:
            raise UnexpectedError(f"Unexpected Error: {response_text}")

    class Chat:
        def __init__(self, api):
            self.api = api
            self.completions = self.Completions(api)

        class Completions:
            def __init__(self, api):
                self.api = api

            async def create(self, messages=None, model="gpt-3.5-turbo", max_tokens=100000, stream=False):
                if messages is None:
                    raise PromptNotProvidedError()

                url = f"{self.api.url}/chat/completions"
                data = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": messages,
                    "stream": stream,
                }

                try:
                    if not stream:
                        async with aiohttp.ClientSession(headers=self.api.headers) as session:
                            if not stream:
                                async with session.post(url, json=data) as response:
                                    response_text = await response.text()
                                    if response.status == 200:
                                        response_json = await response.json()
                                        return self._parse_response(response_json)
                                    else:
                                        self.api.handle_http_error(response.status, response_text)
                    else:
                        return self._process_stream(url, data, self.api.headers)
                   

                except aiohttp.ClientError as e:
                    raise ServerError(f"Connection error: {str(e)}")
                except json.JSONDecodeError:
                    raise UnexpectedError("Invalid JSON response")
                except Exception as e:
                    raise UnexpectedError(f"An unexpected error occurred: {e}")

            def _parse_response(self, response_json):
                """
                Parse a full non-streaming response.
                """
                choices = [
                    Choice(
                        finish_reason=choice.get("finish_reason", None),
                        index=choice.get("index", None),
                        logprobs=choice.get("logprobs", None),
                        message=ChatCompletionMessage(
                            content=choice.get("message", {}).get("content", None),
                            refusal=choice.get("message", {}).get("refusal", None),
                            role=choice.get("message", {}).get("role", None),
                            audio=choice.get("message", {}).get("audio", None),
                            function_call=choice.get("message", {}).get("function_call", None),
                            tool_calls=choice.get("message", {}).get("tool_calls", None),
                        )
                    )
                    for choice in response_json.get("choices", [])
                ]
                usage = CompletionUsage(
                    completion_tokens=response_json.get("usage", {}).get("completion_tokens", None),
                    prompt_tokens=response_json.get("usage", {}).get("prompt_tokens", None),
                    total_tokens=response_json.get("usage", {}).get("total_tokens", None),
                    completion_tokens_details=CompletionTokensDetails(
                        accepted_prediction_tokens=response_json.get("usage", {}).get("completion_tokens_details", {}).get("accepted_prediction_tokens", None),
                        audio_tokens=response_json.get("usage", {}).get("completion_tokens_details", {}).get("audio_tokens", None),
                        reasoning_tokens=response_json.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", None),
                        rejected_prediction_tokens=response_json.get("usage", {}).get("completion_tokens_details", {}).get("rejected_prediction_tokens", None),
                    ),
                    prompt_tokens_details=PromptTokensDetails(
                        audio_tokens=response_json.get("usage", {}).get("prompt_tokens_details", {}).get("audio_tokens", None),
                        cached_tokens=response_json.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", None)
                    )
                )
                return ChatCompletion(
                    id=response_json.get("id", None),
                    choices=choices,
                    created=response_json.get("created", None),
                    model=response_json.get("model", None),
                    object=response_json.get("object", None),
                    service_tier=response_json.get("service_tier", None),
                    system_fingerprint=response_json.get("system_fingerprint", None),
                    usage=usage
                )

            async def _process_stream(self, url, body, headers):
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.post(url, json=body) as response:
                        async for line in response.content:
                            decoded_line = line.decode("utf-8").strip()
                            if decoded_line.startswith("data: "):
                                try:
                                    data_json = json.loads(decoded_line[len("data: "):])
                                    yield self._parse_streaming_chunk(data_json)
                                except json.JSONDecodeError:
                                    continue

            def _parse_streaming_chunk(self, data):
                """
                Parse a single chunk of a streaming response.
                """
                choices = [
                    chat_stream.Choice(
                        delta=chat_stream.ChoiceDelta(
                            content=choice["delta"].get("content"),
                            function_call=choice["delta"].get("function_call"),
                            refusal=choice["delta"].get("refusal"),
                            role=choice["delta"].get("role"),
                            tool_calls=choice["delta"].get("tool_calls")
                        ),
                        finish_reason=choice.get("finish_reason"),
                        index=choice["index"],
                        logprobs=choice.get("logprobs")
                    )
                    for choice in data.get("choices", [])
                ]

                return chat_stream.ChatCompletionChunk(
                    id=data["id"],
                    choices=choices,
                    created=data["created"],
                    model=data["model"],
                    object=data["object"],
                    service_tier=data.get("service_tier"),
                    system_fingerprint=data.get("system_fingerprint"),
                    usage=data.get("usage")
                )

    class Images:
        def __init__(self, api):
            self.api = api

        async def generate(
            self,
            prompt,
            model="dall-e-3",
            size="1024x1024",
            n=1,
        ):
            """
            Generate an image based on the given prompt and model.
            Raises custom exceptions for missing required parameters.
            """
            if not prompt:
                raise PromptNotProvidedError("The 'prompt' parameter is required and cannot be None or empty.")

            url = f"{self.api.url}/images/generations"
            payload = {
                "prompt": prompt,
                "model": model,
                "n": n,
                "size": size,
            }

            try:
                async with aiohttp.ClientSession(headers=self.api.headers) as session:
                    async with session.post(url, json=payload) as response:
                        response_text = await response.text()

                        if response.status == 200:
                            json_response = await response.json()

                            # Parse the JSON into the ImagesResponse dataclass
                            parsed_response = ImagesResponse(
                                created=json_response["created"],
                                data=[
                                    Image(
                                        url=item.get("url"),
                                        b64_json=item.get("b64_json"),
                                        revised_prompt=item.get("revised_prompt")
                                    )
                                    for item in json_response.get("data", [])
                                ]
                            )
                            return parsed_response
                        else:
                            raise APIError(f"Unexpected status code: {response.status}")

            except aiohttp.ClientResponseError as http_err:
                if http_err.status == 401:
                    raise APIError("Invalid Key") from http_err
                elif http_err.status == 403:
                    raise APIError("Forbidden") from http_err
                elif http_err.status == 404:
                    raise APIError("The URL was not found.") from http_err
                elif http_err.status == 500:
                    raise APIError("Internal server error.") from http_err
                else:
                    raise APIError(f"HTTP error occurred with status code {http_err.status}.") from http_err

            except aiohttp.ClientConnectionError:
                raise ConnectionError("Connection failed.")
            except aiohttp.ClientTimeout:
                raise TimeoutError("Request timed out.")
            except Exception as ex:
                raise APIError("Unknown error occurred.") from ex


    class Audio:
        def __init__(self, api):
            self.api = api
            self.transcriptions = self.Transcriptions(api)

        class Transcriptions:
            def __init__(self, api):
                self.api = api

            async def create(
                self,
                audio_file,
                model: str = "whisper-1",
            ) -> Optional[TranscriptionResponse]:
                """
                Transcribe or translate an audio file using OpenAI's Whisper API.

                Args:
                    audio_file (file-like object): The audio file to transcribe.
                    model (str): The model ID to use. Default is "whisper-1".

                Returns:
                    TranscriptionResponse: The transcription result.
                """
                if audio_file is None:
                    raise ValueError("The 'audio_file' parameter is required and cannot be None.")

                url = f"{self.api.url}/audio/translations"
                headers = {
                    "Authorization": f"Bearer {self.api.api_key}"
                }

                data = aiohttp.FormData()
                data.add_field("audio", audio_file, filename="audio.wav")
                data.add_field("model", model)

                try:
                    async with aiohttp.ClientSession(headers=headers) as session:
                        async with session.post(url, data=data) as response:
                            response_text = await response.text()

                            if response.status == 200:
                                json_response = await response.json()

                                # Parse into TranscriptionResponse dataclass
                                transcription = TranscriptionResponse(
                                    text=json_response.get("text", "")
                                )

                                # Handle optional fields if present
                                if "language" in json_response:
                                    transcription.language = json_response["language"]
                                if "duration" in json_response:
                                    transcription.duration = json_response["duration"]
                                if "words" in json_response:
                                    transcription.words = [
                                        Word(
                                            word=word.get("word", ""),
                                            start=word.get("start", 0.0),
                                            end=word.get("end", 0.0)
                                        ) for word in json_response.get("words", [])
                                    ]
                                if "segments" in json_response:
                                    transcription.segments = [
                                        Segment(
                                            id=segment.get("id", 0),
                                            seek=segment.get("seek", 0.0),
                                            start=segment.get("start", 0.0),
                                            end=segment.get("end", 0.0),
                                            text=segment.get("text", ""),
                                            tokens=segment.get("tokens", []),
                                            temperature=segment.get("temperature", 0.0),
                                            avg_logprob=segment.get("avg_logprob", 0.0),
                                            compression_ratio=segment.get("compression_ratio", 0.0),
                                            no_speech_prob=segment.get("no_speech_prob", 0.0)
                                        ) for segment in json_response.get("segments", [])
                                    ]

                                return transcription

                except aiohttp.ClientResponseError as http_err:
                    if http_err.status == 401:
                        raise APIError("Unauthorized access. Please check your API key or credentials.") from http_err
                    elif http_err.status == 403:
                        raise APIError("Forbidden. You might not have permissions to access this resource.") from http_err
                    elif http_err.status == 404:
                        raise APIError("The requested endpoint was not found. Check the API URL.") from http_err
                    elif http_err.status == 429:
                        raise APIError("Too many requests. You have exceeded your rate limit.") from http_err
                    elif http_err.status >= 500:
                        raise APIError("Internal server error at the API side.") from http_err
                    else:
                        raise APIError(f"HTTP error occurred with status code {http_err.status}.") from http_err

                except aiohttp.ClientConnectionError:
                    raise APIError("Failed to connect to the API. Please check your network connection.")
                except aiohttp.ClientTimeout:
                    raise APIError("The request timed out. Try again later or increase the timeout.")
                except ValueError:
                    raise APIError("Failed to parse the API response as JSON.")
                except Exception as ex:
                    raise APIError("An unknown error occurred while processing the request.") from ex

    class embeddings:
        def __init__(self, api):
            self.api = api

        async def create(self, input=None, model="text-embedding-ada-002"):
            """
            Generate text embeddings based on the given text and model.
            Raises custom exceptions for missing required parameters.
            """
            if input is None:
                raise PromptNotProvidedError("The 'input' parameter is required and cannot be None.")

            url = f"{self.api.url}/embeddings"
            payload = {
                "text": input,
                "model": model
            }

            async with aiohttp.ClientSession(headers=self.api.headers) as session:
                try:
                    async with session.post(url, json=payload) as response:
                        if response.status == 401:
                            raise APIError("Unauthorized access. Please check your API key.")
                        elif response.status == 403:
                            raise APIError("Forbidden. You might not have permission to access this resource.")
                        elif response.status == 404:
                            raise APIError("The requested endpoint was not found. Check the API URL.")
                        elif response.status == 429:
                            raise APIError("Too many requests. You have exceeded your rate limit.")
                        elif response.status >= 500:
                            raise APIError("Internal server error at the API side.")
                        elif not response.ok:
                            raise APIError(f"HTTP error occurred with status code {response.status}.")

                        data = await response.json()
                        return CreateEmbeddingResponse(
                            data=[Embedding(**d) for d in data["data"]],
                            model=data["model"],
                            usage=Usage(**data["usage"]) if "usage" in data else None
                        )

                except aiohttp.ClientError as e:
                    raise ConnectionError(f"Failed to connect to the API: {e}") from e
                except asyncio.TimeoutError:
                    raise TimeoutError("The request timed out. Try again later or increase the timeout.")
                except ValueError as json_err:
                    raise APIError("Failed to parse the API response as JSON.") from json_err
                except Exception as ex:
                    raise APIError("An unknown error occurred while processing the request.") from ex
