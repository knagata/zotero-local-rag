# src/docling_worker.py
"""Run Docling extraction in an isolated, persistent subprocess.

Background: Docling's internal RapidOCR dependency has a known semaphore leak
on macOS that can eventually crash the process running it. Docling previously
ran in-process inside index_from_zotero.py -- the same process that holds the
live Chroma client and performs col.upsert() calls. A Docling/RapidOCR crash
landing mid-write there can permanently corrupt the Chroma HNSW segment (the
vector index falls behind the embeddings_queue and every subsequent
count()/get()/query() fails).

This module moves Docling extraction into a separate child process so that if
Docling/RapidOCR crashes, it can only take down that worker subprocess -- never
the parent process that owns the Chroma client. Docling's model loading is
expensive (seconds), so a single worker process is kept alive for the whole
indexing run and is only respawned if it dies.

The child is started with the ``spawn`` start method explicitly (never
``fork``): fork is unsafe with native-code deps like torch/RapidOCR on macOS
and is plausibly part of why the leak/crash happens at all.
"""
from __future__ import annotations

import gc
import multiprocessing
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _relieve_worker_memory_pressure() -> None:
    """Release safe caches without touching the macOS MPS allocator.

    Calling ``torch.mps.empty_cache()`` after Docling/RapidOCR extraction can
    segfault inside ``MPSAllocator/MPSStream`` on macOS.  A hard crash here
    discards an otherwise complete OCR result before it crosses the worker
    pipe.  The isolated worker is short-lived and the OS reclaims its MPS
    allocations at exit, so stability takes precedence over eager MPS cache
    release. CUDA cleanup remains useful and has no bearing on that crash.
    """
    try:
        gc.collect()
    except Exception:
        pass
    if (os.environ.get("TORCH_EMPTY_CACHE") or "1") != "1":
        return
    try:
        import torch  # type: ignore

        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _worker_loop(conn) -> None:
    """Child process entry point.

    Must be a module-level function (not a closure/lambda): the ``spawn``
    start method pickles the target, and closures/lambdas aren't picklable.

    Reads ``(command, ...)`` messages off ``conn`` in a loop:
      - ("extract", pdf_path_str, attachment_key, meta_base): run Docling
        extraction and reply with ("ok", chunks, quality_info) on success or
        ("error", message) on a normal (non-crashing) exception, then keep
        looping -- a single failed file must not end the worker.
      - ("shutdown",): stop the loop and let the process exit.

    If Docling/RapidOCR hard-crashes the process (e.g. a segfault from the
    semaphore leak), this function never gets to send a reply at all; the
    parent detects that via a broken pipe / EOF, not via anything this
    function does.
    """
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            # Parent closed its end of the pipe (e.g. during shutdown/GC).
            break
        except Exception:
            break

        if not msg:
            continue

        kind = msg[0]
        if kind == "shutdown":
            break

        if kind == "extract":
            _, pdf_path_str, attachment_key, meta_base = msg
            try:
                # Imported lazily (function-local, not module top-level) for
                # two reasons: (1) it avoids a circular import at module load
                # time, since index_from_zotero.py imports DoclingWorker from
                # this module; (2) docling itself is only imported inside
                # extract_chunks_from_pdf_with_docling, so importing the
                # docling_extract module here is cheap -- the expensive model
                # load happens on the first actual extract() call, inside
                # this child process, and is paid once per worker lifetime.
                from docling_extract import extract_chunks_from_pdf_with_docling

                chunks, quality_info = extract_chunks_from_pdf_with_docling(
                    Path(pdf_path_str), attachment_key, meta_base,
                )
                # The worker (not the parent) now holds the torch/MPS state,
                # so the cache-clearing cleanup belongs here.
                _relieve_worker_memory_pressure()
                conn.send(("ok", chunks, quality_info))
            except Exception as exc:  # noqa: BLE001 - report, don't crash the worker
                try:
                    conn.send(("error", str(exc)))
                except Exception:
                    # Pipe is gone; nothing more we can do -- exit the loop.
                    break
            continue


class DoclingWorker:
    """Owns one persistent, isolated Docling subprocess for an indexing run.

    Usage: create one instance per indexing run (not per file), call
    ``extract()`` for each PDF, and call ``shutdown()`` when the run ends.

    If the subprocess crashes or hangs, ``extract()`` raises ``RuntimeError``
    for that one file and transparently respawns a fresh subprocess so the
    next call can proceed. The caller (index_from_zotero.py) treats that
    ``RuntimeError`` exactly like any other extraction failure -- mark the
    attachment failed/retryable and continue with the next file. The parent
    process itself never crashes or needs restarting.
    """

    def __init__(self, timeout_sec: float = 3600):
        self.timeout_sec = float(timeout_sec)
        self._ctx = multiprocessing.get_context("spawn")
        self._parent_conn = None
        self._process = None
        self._start()

    def _start(self) -> None:
        """Spawn a fresh worker process and Pipe, replacing any previous one."""
        old_conn = self._parent_conn
        parent_conn, child_conn = self._ctx.Pipe()
        process = self._ctx.Process(target=_worker_loop, args=(child_conn,), daemon=True)
        process.start()
        # The parent doesn't need its copy of the child's end once the child
        # process owns it.
        child_conn.close()
        if old_conn is not None:
            try:
                old_conn.close()
            except Exception:
                pass
        self._parent_conn = parent_conn
        self._process = process

    def extract(
        self, pdf_path: Path, attachment_key: str, meta_base: dict
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Extract chunks for one PDF via the persistent worker subprocess.

        Returns (chunks, quality_info) on success.

        Raises RuntimeError if:
          - the worker timed out (subprocess is terminated and respawned), or
          - the worker crashed while processing this file (respawned), or
          - extraction raised a normal (non-crashing) exception inside the
            worker (worker is left running as-is; only this file failed).
        """
        if self._process is None or not self._process.is_alive():
            # Crashed since the last call; respawn before proceeding.
            self._start()

        try:
            self._parent_conn.send(("extract", str(pdf_path), attachment_key, meta_base))
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._start()
            raise RuntimeError(
                f"Docling worker crashed while processing: {pdf_path}"
            ) from exc

        if not self._parent_conn.poll(self.timeout_sec):
            # Hung. Kill the presumably-stuck worker and respawn for next time.
            try:
                self._process.terminate()
                self._process.join(timeout=5)
            except Exception:
                pass
            self._start()
            raise RuntimeError(f"Docling worker timed out after {self.timeout_sec}s: {pdf_path}")

        try:
            response = self._parent_conn.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            self._start()
            raise RuntimeError(
                f"Docling worker crashed while processing: {pdf_path}"
            ) from exc

        if self._process.exitcode is not None:
            # The process ended (e.g. crashed right after/while replying).
            # Treat this the same as a mid-extraction crash.
            self._start()
            raise RuntimeError(f"Docling worker crashed while processing: {pdf_path}")

        kind = response[0]
        if kind == "ok":
            _, chunks, quality_info = response
            return chunks, quality_info
        if kind == "error":
            _, message = response
            # The worker itself is alive and fine -- only this file failed --
            # so do not respawn.
            raise RuntimeError(message)
        raise RuntimeError(
            f"Docling worker returned an unexpected response for {pdf_path}: {response!r}"
        )

    def shutdown(self) -> None:
        """Stop the worker cleanly. Safe to call multiple times or after a crash."""
        process = self._process
        if process is None:
            return
        conn = self._parent_conn
        try:
            if process.is_alive():
                try:
                    conn.send(("shutdown",))
                except Exception:
                    pass
                process.join(timeout=5)
        except Exception:
            pass
        try:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        except Exception:
            pass
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass
        self._process = None
        self._parent_conn = None
