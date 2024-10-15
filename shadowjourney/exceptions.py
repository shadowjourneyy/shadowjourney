# exceptions.py

class APIError(Exception):
    """Custom exception for general API errors."""
    pass

class InvalidRequestError(APIError):
    """Raised when the API request is invalid (e.g., missing parameters)."""
    pass

class AuthenticationError(APIError):
    """Raised when authentication fails (e.g., invalid API key)."""
    pass

class AccessDeniedError(APIError):
    """Raised when access is denied (e.g., insufficient permissions)."""
    pass

class ResourceNotFoundError(APIError):
    """Raised when the requested resource is not found (404 error)."""
    pass

class ServerError(APIError):
    """Raised when the server encounters an internal error (500 error)."""
    pass

class PromptNotProvidedError(InvalidRequestError):
    """Raised when a required prompt is not provided in a request."""
    def __init__(self, message="Prompt not provided."):
        super().__init__(message)

class ModelNotProvidedError(InvalidRequestError):
    """Raised when a required model is not provided in a request."""
    def __init__(self, message="Model not provided."):
        super().__init__(message)

class UnexpectedError(APIError):
    """Raised when an unexpected error occurs (any unhandled status code)."""
    pass
