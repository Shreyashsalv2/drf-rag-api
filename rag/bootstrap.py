"""Cold-start setup for the Vercel demo: create the SQLite tables and load the sample documents."""

import threading

from django.core.management import call_command

_done = False
_lock = threading.Lock()


def prepare() -> None:
    global _done
    with _lock:
        if _done:
            return
        call_command("migrate", interactive=False, verbosity=0)
        call_command("load_samples", verbosity=0)
        _done = True
