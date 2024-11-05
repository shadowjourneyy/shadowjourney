import requests
from urllib.parse import urlparse
from .exceptions import APIError, InvalidRequestError, AuthenticationError, AccessDeniedError, \
    ResourceNotFoundError, ServerError, PromptNotProvidedError, ModelNotProvidedError, UnexpectedError

class API:
    """
    The main client for interacting with the ShadowJourney API.
    """
    def __init__(self, key, baseurl="https://shadowjourney.xyz/v1"):
        """
        Initialize the API client with an API key and an optional base URL.
        Validates the base URL.
        """
        self.api_key = key
        self.base_url = baseurl if self.validate_url(baseurl) else "https://shadowjourney.xyz/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def validate_url(self, url):
        """
        Validates the URL format.
        """
        parsed_url = urlparse(url)
        return all([parsed_url.scheme, parsed_url.netloc])

    def chatcmpl(self, model="gpt-3.5-turbo", max_tokens=100, prompt=None, system_prompt="ai", stream=False, json=False):
        """
        Send a chat completion request to the v1 API with an optional system prompt.
        Raises custom exceptions for missing required parameters.
        """
        if prompt is None:
            raise PromptNotProvidedError()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        url = f"{self.base_url}/chat/completions"
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "stream": stream,
        }

        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            if json:
                return response.json()
            else:
                return response.content.decode('utf-8')
        except requests.exceptions.HTTPError as e:
            self.handle_http_error(e, response)
        except requests.exceptions.RequestException as e:
            print(e)
            return None

    def list_models(self):
        """
        Fetch the list of available models from the API.
        """
        url = f"{self.base_url}/models"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            self.handle_http_error(e, response)
        except requests.exceptions.RequestException as e:
            print(e)
            return None

    def GenImg(self, prompt=None, model="dalle-3", size="1024x1024", n=1):
        """
        Generate an image based on the given prompt and model.
        Raises custom exceptions for missing required parameters.
        """
        if prompt is None:
            raise PromptNotProvidedError()

        url = f"{self.base_url}/images/generations"
        payload = {
            "prompt": prompt,
            "model": model,
            "n": n,
            "size": size
        }

        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            self.handle_http_error(e, response)
        except requests.exceptions.RequestException as e:
            print(e)
            return None

    def embeddings(self, text=None, model="text-embedding-ada-002"):
        """
        Generate text embeddings based on the given text and model.
        Raises custom exceptions for missing required parameters.
        """
        if text is None:
            raise PromptNotProvidedError()

        url = f"{self.base_url}/embeddings"
        payload = {
            "text": text,
            "model": model
        }

        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            self.handle_http_error(e, response)
        except requests.exceptions.RequestException as e:
            print(e)
            return None

    def transcriptions(self, audio_file=None, model="whisper-1"):
        """
        Generate text transcription from the given audio file and model.
        Raises custom exceptions for missing required parameters.
        """
        if audio_file is None:
            raise PromptNotProvidedError()

        url = f"{self.base_url}/transcriptions"
        files = {'audio': audio_file}
        payload = {
            "model": model
        }

        try:
            response = requests.post(url, files=files, data=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            self.handle_http_error(e, response)
        except requests.exceptions.RequestException as e:
            print(e)
            return None

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
        elif status_code == 500:
            raise ServerError(f"Server Error: {response.text}")
        else:
            raise UnexpectedError(f"Unexpected Error: {response.text}")