"""Ruth CLI entrypoint (the native inference engine is not yet qualified)."""
from ruth_runtime.launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
