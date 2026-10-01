class CheckpointError(Exception):
    """Base class for all checkpoint-package errors."""


class CheckpointWriteError(CheckpointError):
    """Raised when one or more chunk/object writes failed or failed
    post-write validation during save_checkpoint(). The checkpoint is left
    marked INCOMPLETE (or, if the failure happened before the manifest was
    written, simply doesn't exist as far as the catalog is concerned) --
    callers must never see a successful return alongside a failed write.
    """


class CheckpointNotFoundError(CheckpointError):
    """No catalog entry (or, for load-by-id, no manifest) for the requested
    checkpoint id."""


class CheckpointIncompleteError(CheckpointError):
    """The requested checkpoint's manifest exists but status != COMPLETE."""


class CheckpointCorruptError(CheckpointError):
    """verify() found a checksum/size mismatch or a missing chunk/object."""
