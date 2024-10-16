import hashlib
import pathlib
import json
from joblib import Memory

class HashedMemory(Memory):

    def __init__(self, location, verbose=0):
        super().__init__(location=location, verbose=verbose)

    def _hash(self, string):
        """Returns an MD5 hash for the given string."""
        return hashlib.md5(string.encode()).hexdigest()

    def cache(self, func):
        """Decorator that caches the function's result using a hash of the arguments."""
        def wrapper(*args, **kwargs):
            # Serialize arguments to create a unique key
            args_str = json.dumps((args, kwargs), sort_keys=True, default=str)
            key = self._hash(args_str)
            cache_dir = pathlib.Path(self.location) / key
            cache_file = cache_dir / f"{func.__name__}_result.json"
            
            # If the result is already cached, load it
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    result = json.load(f)
                return result
            
            # Otherwise, compute the result and cache it
            result = func(*args, **kwargs)
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            
            return result
        return wrapper
