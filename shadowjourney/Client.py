import requests
from urllib.parse import urlparse
from .exceptions import *
from .responses.chat import ChatCompletion, Choice, ChatCompletionMessage, CompletionUsage, CompletionTokensDetails, PromptTokensDetails
from .responses import chat_stream
from .responses.images import *
from .responses.transcription import *
from .responses.embedding import *
import json

class API:
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

    def handle_http_error(self, e, response):
        """
        Handles HTTP errors and raises custom exceptions based on status codes.
        """
        status_code = response.status_code if response else None
        if status_code == 400:
            raise InvalidRequestError(f"Bad Request: {response.text}")
        elif status_code == 401:
            raise AuthenticationError(f"Authentication Failed: {response.text}")
        elif status_code == 403:
            raise AccessDeniedError(f"Access Denied: {response.text}")
        elif status_code == 404:
            raise ResourceNotFoundError(f"Resource Not Found: {response.text}")
        elif status_code == 429:
            raise RateLimitError(f"Rate limit reached!: {response.text}")
        elif status_code == 500:
            raise ServerError(f"Server Error: {response.text}")
        else:
            raise UnexpectedError(f"Unexpected Error: {response.text}")

    class Chat:
        def __init__(self, api):
            self.api = api
            self.completions = self.Completions(api)

        class Completions:
            def __init__(self, api):
                self.api = api

            def create(self, messages=None, model="gpt-3.5-turbo", max_tokens=100000, stream=False):
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
                        response = requests.post(url, headers=self.api.headers, json=data)
                        response.raise_for_status()
                        response_json = response.json()
                        
                        if response.ok:
                            return self._parse_response(response_json)
                        else:
                            # Handle unexpected non-2xx responses
                            raise UnexpectedError(f"Unexpected status code: {response.status_code}")
                    else:
                        return self._process_stream(url, headers=self.api.headers, body=data)

                except requests.exceptions.HTTPError as e:
                    status_code = e.response.status_code if e.response else None
                    if status_code == 400:
                        raise InvalidRequestError(f"Invalid request.")
                    elif status_code == 401:
                        raise AuthenticationError(f"Authentication failed.")
                    elif status_code == 403:
                        raise AccessDeniedError(f"Access denied. Please Check your api key")
                    elif status_code == 404:
                        raise ResourceNotFoundError(f"Resource not found")
                    elif status_code == 429:
                        raise RateLimitError(f"Rate limit exceede")
                    elif status_code >= 500:
                        raise ServerError(f"{e}")
                    else:
                        raise UnexpectedError(f"HTTP error: {e}")

                except requests.exceptions.ConnectionError as e:
                    raise ServerError(f"Connection error.")
                except requests.exceptions.Timeout as e:
                    raise ServerError(f"Request timed out.")
                except json.JSONDecodeError as e:
                    raise UnexpectedError(f"Invalid JSON response")
                except APIError as e:
                    raise e
                except Exception as e:
                    raise UnexpectedError(f"An unexpected error occurred: {e}")

            def _parse_response(self, response_json):
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

            def _parse_streaming_chunk(self, data):
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
                    ) for choice in data.get("choices", [])
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

            def _process_stream(self, url, headers, body):
                response = requests.post(url, headers=headers, json=body, stream=True)
                for line in response.iter_lines(decode_unicode=True):
                    if line and line.startswith("data: "):
                        try:
                            data_json = json.loads(line[len("data: "):])
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
                                ) for choice in data_json.get("choices", [])
                            ]

                            yield chat_stream.ChatCompletionChunk(
                                id=data_json["id"],
                                choices=choices,
                                created=data_json["created"],
                                model=data_json["model"],
                                object=data_json["object"],
                                service_tier=data_json.get("service_tier"),
                                system_fingerprint=data_json.get("system_fingerprint"),
                                usage=data_json.get("usage")
                            )
                        except json.JSONDecodeError as e:
                            continue  

    class Images:
        def __init__(self, api):
            self.api = api

        def generate(
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
                "size": size
            }

            try:
                # Make the POST request
                response = requests.post(url, json=payload, headers=self.api.headers)

                # Raise HTTPError for 4xx/5xx responses
                response.raise_for_status()

                # Parse the response
                json_response = response.json()

                if response.ok:
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
                    raise APIError(f"Unexpected status code: {response.status_code}")

            except requests.exceptions.HTTPError as http_err:
                if response.status_code == 401:
                    raise APIError("Invalid Key") from http_err
                elif response.status_code == 403:
                    raise APIError("Forbidden") from http_err
                elif response.status_code == 404:
                    raise APIError("The url was not found.") from http_err
                elif response.status_code == 500:
                    raise APIError("Internal server error.") from http_err
                else:
                    raise APIError(f"HTTP error occurred with status code {response.status_code}.") from http_err
            except requests.exceptions.ConnectionError as conn_err:
                raise ConnectionError("Connection failed.") from conn_err
            except requests.exceptions.Timeout as timeout_err:
                raise TimeoutError("Request timed out.") from timeout_err
            except requests.exceptions.RequestException as req_err:
                raise APIError("API request failed.") from req_err
            except ValueError as json_err:
                raise ValueError("Invalid JSON response.") from json_err
            except Exception as ex:
                raise APIError("Unknown error occurred.") from ex
            
    class audio:
        def __init__(self, api):
            self.api = api
            self.transcriptions = transcriptions(api)

            class transcriptions:
                def __init__(self, api):
                    self.api = api

                def create(
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
                    
                    Raises:
                        PromptNotProvidedError: If the prompt is required but not provided.
                        APIError: If an error occurs during the API request.
                    """
                    if audio_file is None:
                        raise ValueError("The 'audio_file' parameter is required and cannot be None.")

                    url = f"{self.api.url}/audio/translations"
                    headers = {
                        "Authorization": f"Bearer {self.api.api_key}"
                    }

                    files = {
                        "audio": audio_file
                    }
                    data = {
                        "model": model,
                    }

                    # Remove None values to avoid sending them in the request
                    data = {k: v for k, v in data.items() if v is not None}
                    try:
                        # Make the POST request
                        response = requests.post(url, headers=headers, data=data, files=files)

                        # Raise HTTPError for bad responses (4xx and 5xx)
                        response.raise_for_status()

                        # Parse the response based on the response_format
                        if response.status_code == response.ok:
                            json_response = response.json()

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

                    except requests.exceptions.HTTPError as http_err:
                        if response.status_code == 401:
                            raise APIError("Unauthorized access. Please check your API key or credentials.") from http_err
                        elif response.status_code == 403:
                            raise APIError("Forbidden. You might not have permissions to access this resource.") from http_err
                        elif response.status_code == 404:
                            raise APIError("The requested endpoint was not found. Check the API URL.") from http_err
                        elif response.status_code == 429:
                            raise APIError("Too many requests. You have exceeded your rate limit.") from http_err
                        elif response.status_code >= 500:
                            raise APIError("Internal server error at the API side.") from http_err
                        else:
                            raise APIError(f"HTTP error occurred with status code {response.status_code}.") from http_err

                    except requests.exceptions.ConnectionError as conn_err:
                        raise APIError("Failed to connect to the API. Please check your network connection.") from conn_err
                    except requests.exceptions.Timeout as timeout_err:
                        raise APIError("The request timed out. Try again later or increase the timeout.") from timeout_err
                    except requests.exceptions.RequestException as req_err:
                        raise APIError("An unexpected error occurred while making the API request.") from req_err
                    except ValueError as json_err:
                        raise APIError("Failed to parse the API response as JSON.") from json_err
                    except KeyError as key_err:
                        raise APIError("The response JSON is missing expected keys.") from key_err
                    except Exception as ex:
                        raise APIError("An unknown error occurred while processing the request.") from ex
                    

            class embeddings:
                def __init__(self, api):
                    self.api = api
                    self.embeddings = embeddings(api)

                def create(self, input=None, model="text-embedding-ada-002"):
                    """
                    Generate text embeddings based on the given text and model.
                    Raises custom exceptions for missing required parameters.
                    """
                    if input is None:
                        raise PromptNotProvidedError()

                    url = f"{self.api.url}/embeddings"
                    payload = {
                        "text": input,
                        "model": model
                    }

                    try:
                        response = requests.post(url, json=payload, headers=self.api.headers)
                        response.raise_for_status()
                        data = response.json()
                        return CreateEmbeddingResponse(
                            data=[Embedding(**d) for d in data["data"]],
                            model=data["model"],
                            usage=Usage(**data["usage"]) if data["usage"] else None
                        )
                    except requests.exceptions.HTTPError as e:
                        self.handle_http_error(e, response)
                    except requests.exceptions.RequestException as e:
                        raise APIError(e)    
                    
