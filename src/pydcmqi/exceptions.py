class DcmqiError(RuntimeError):
    """Raised when a dcmqi CLI tool fails."""

    def __init__(self, cmd: list[str], returncode: int, stderr: str) -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        tool = cmd[0] if cmd else "dcmqi"
        super().__init__(f"{tool} failed (exit code {returncode}):\n{stderr.strip()}")
