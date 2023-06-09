# SPDX-FileCopyrightText: 2020-2023 CERN
# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Provide `catching_exceptions()` for use in runner threads."""

from __future__ import annotations

import contextlib
import logging
import sys
import typing as t
from traceback import TracebackException

from cernml.coi import cancellation


class BenignCancelledError(cancellation.CancelledError):
    """Cancellation error that we raise, not the optimization problem."""


@contextlib.contextmanager
def catching_exceptions(
    name: str,
    logger: logging.Logger,
    *,
    token_source: cancellation.TokenSource,
    on_success: t.Callable[[], t.Any],
    on_cancel: t.Callable[[], t.Any],
    on_exception: t.Callable[[TracebackException], t.Any],
) -> t.Iterator[None]:
    """Context manager that turns exceptions into callbacks.

    This is used by jobs to catch *all* exceptions and emit them via Qt
    signals instead. This is necessary because exiting a thread via
    exception takes down the entire application.

    This also takes care of most of the token source lifecycle
    management for us.

    Args:
        name: A string identifying the process happening in this
            context. This appears  only in logging messages.
        logger: The logger where the outcome (success, cancellation,
            exception) should be logged.
        token_source: A token source that should be reset if possible.
        on_success: Called if the context is exited without exception.
        on_cancel: Called if the context is exited via
            :exc:`~cancellation.CancelledError`.
        on_exception: Called if the context is left through *any* other
            exception. The argument is a :exc:`TracebackException` with
            as much information as possible without including local
            variables.
    """
    # pylint: disable = bare-except
    try:
        yield
        logger.info(f"finished {name}")
        on_success()
        # Catch weird race conditions: If we successfully run through
        # and a cancellation arrives _just_ after, we automatically
        # complete it.
        if token_source.token.cancellation_requested:
            token_source.token.complete_cancellation()
            token_source.reset_cancellation()
    except BenignCancelledError:
        logger.info(f"cancelled {name}")
        token_source.token.complete_cancellation()
        token_source.reset_cancellation()
        on_cancel()
    except cancellation.CancelledError:
        logger.info(f"cancelled {name}")
        if token_source.can_reset_cancellation:
            token_source.reset_cancellation()
        on_cancel()
    except:
        logger.error(f"aborted {name}", exc_info=True)
        on_exception(TracebackException(*sys.exc_info()))
